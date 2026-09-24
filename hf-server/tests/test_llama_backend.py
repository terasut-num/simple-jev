"""Numerical and HTTP checks for the llama.cpp GGUF backend.

Two layers of coverage:

1. Orchestration tests (stubbed llama_cpp): a fake engine records every batch,
   memory call, and logit read. They verify the shared-prefix flow itself —
   prefix prefill on sequence 0, per-row sequence copies, packed rows with
   per-row final-position logits, sequence cleanup between batches, metrics,
   and cooperative cancellation — without loading any weights.

2. Real-engine tests (skipped unless llama-cpp-python is installed and
   SIMPLE_JEV_GGUF points at a local GGUF): they compare the backend's
   shared-prefix scores against independent full-prompt decodes on the same
   context, plus an end-to-end HTTP check. These establish execution
   equivalence, not answer quality. The Vulkan backend of some builds is
   less precise than CPU; pass SIMPLE_JEV_DEVICE=cpu to pin the comparison.
"""

import asyncio
import importlib.util
import os
import sys
import types

import pytest

import llama_cpp  # noqa: F401  (presence gates this module)
from llama_cpp import _internals

from hf_server import (
    Branch,
    CompiledRequest,
    LlamaCppBackend,
    PromptCompiler,
)

from common import prepare_prompt

GGUF = os.environ.get("SIMPLE_JEV_GGUF")
DEVICE = os.environ.get("SIMPLE_JEV_DEVICE", "cpu")

real_engine = pytest.mark.skipif(
    GGUF is None, reason="Set SIMPLE_JEV_GGUF to a local GGUF file"
)


class StubBatch:
    def __init__(self, n_tokens, embd, n_seq_max):
        self.n_tokens = 0
        self.token = [0] * n_tokens
        self.pos = [0] * n_tokens
        self.seq_id = [[0] * n_seq_max for _ in range(n_tokens)]
        self.n_seq_id = [0] * n_tokens
        self.logits = [False] * n_tokens


class StubContext:
    def __init__(self, n_ctx, calls):
        self.ctx = "stub-ctx"
        self.memory = "stub-memory"
        self.n_ctx_value = n_ctx
        self.calls = calls

    def n_ctx(self):
        return self.n_ctx_value


def install_stub(monkeypatch, *, n_ctx=512, decode_rc=0, logit_value=0.5):
    """Install a recording fake llama_cpp module into sys.modules.

    The backend imports llama_cpp lazily inside _score, so the stub module is
    found without the real package. Every engine call is appended to `calls`
    as a tag plus payload; logits read at position i are [i] + logit_value.
    """
    calls = []
    batches = {}

    def llama_batch_init(n_tokens, embd, n_seq_max):
        batch = StubBatch(n_tokens, embd, n_seq_max)
        batches[id(batch)] = batch
        return batch

    def llama_batch_free(batch):
        batches.pop(id(batch), None)

    def llama_decode(ctx, batch):
        calls.append(
            (
                "decode",
                [
                    (
                        batch.seq_id[j][0],
                        batch.pos[j],
                        batch.token[j],
                        batch.logits[j],
                    )
                    for j in range(batch.n_tokens)
                ],
            )
        )
        return decode_rc

    def llama_get_logits_ith(ctx, i):
        calls.append(("logits", i))
        # Large enough for any token id the byte tokenizer produces.
        return [float(i) + logit_value] * 65536

    def llama_memory_clear(memory, data):
        calls.append(("clear", data))

    def llama_memory_seq_cp(memory, src, dst, p0, p1):
        calls.append(("seq_cp", src, dst, p0, p1))

    def llama_memory_seq_rm(memory, seq, p0, p1):
        calls.append(("seq_rm", seq, p0, p1))

    stub = types.SimpleNamespace(
        llama_batch_init=llama_batch_init,
        llama_batch_free=llama_batch_free,
        llama_decode=llama_decode,
        llama_get_logits_ith=llama_get_logits_ith,
        llama_get_memory=lambda ctx: "stub-memory",
        llama_memory_clear=llama_memory_clear,
        llama_memory_seq_cp=llama_memory_seq_cp,
        llama_memory_seq_rm=llama_memory_seq_rm,
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    context = StubContext(n_ctx, calls)
    return context, calls


def compiled_request():
    """One real plan with three noul questions sharing a long token prefix."""
    body = {
        "model": "m",
        "state": "The bicycle is red and parked outside.",
        "questions": {
            key: {"type": "noul", "instructions": f"Question {key}?"}
            for key in ["a", "b", "c"]
        },
    }
    plan = prepare_prompt(body)

    class ByteTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf8"))

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages) + "\nA:"

    compiled = PromptCompiler(ByteTokenizer()).compile(body)
    assert compiled.plan == plan
    return compiled


def backend(context, **kwargs):
    return LlamaCppBackend(None, context, **kwargs)


async def test_prefix_once_then_packed_batches_with_cleanup(monkeypatch):
    """One prefill decode on seq 0, per-row seq_cp, packed rows, seq_rm cleanup."""
    # The full v1 template compiles to over a thousand tokens per branch with
    # the byte tokenizer; a large stub pool keeps all rows in one suffix batch.
    context, calls = install_stub(monkeypatch, n_ctx=8192)
    compiled = compiled_request()
    sequences = [b.token_ids for b in compiled.branches]
    # The three byte-tokenized branches share the template's long common text;
    # the backend reserves exactly one suffix token per branch after that.
    prefix = 0
    for other in sequences[1:]:
        for i, (left, right) in enumerate(zip(sequences[0], other)):
            if left != right:
                break
            prefix = max(prefix, i + 1)
    expected_prefix = min(prefix, min(map(len, sequences)) - 1)
    result = await backend(context).score(compiled)

    decodes = [call[1] for call in calls if call[0] == "decode"]
    assert len(decodes) == 2, calls
    # First decode: the shared prefix alone on sequence 0, no logits wanted.
    assert len(decodes[0]) == expected_prefix
    assert result.metrics["prefix_tokens"] == expected_prefix
    assert all(seq == 0 and want is False for seq, _, _, want in decodes[0])
    # Second decode: three packed rows on seqs 1..3, one flagged position each.
    row_seqs = [seq for seq, _, _, _ in decodes[1]]
    assert set(row_seqs) == {1, 2, 3}
    flagged = [(seq, pos) for seq, pos, _, want in decodes[1] if want]
    assert len(flagged) == 3
    # Each row's flagged position is its branch's final token position.
    assert [pos for _, pos in flagged] == [len(s) - 1 for s in sequences]
    # Sequence copies cover exactly the prefix; each row is removed after use.
    copies = [(call[1], call[2]) for call in calls if call[0] == "seq_cp"]
    assert copies == [(0, 1), (0, 2), (0, 3)]
    removals = [call[1] for call in calls if call[0] == "seq_rm"]
    assert removals == [1, 2, 3]
    assert ("clear", True) in calls
    metrics = result.metrics
    assert metrics["backend"] == "llama-cpp"
    assert metrics["engine_forwards"] == 2
    assert metrics["prefix_tokens"] == expected_prefix
    assert metrics["suffix_batch_sizes"] == [3]
    assert metrics["scored_positions"] == 3
    assert metrics["logical_prefill_tokens"] == metrics["computed_prompt_tokens"]


async def test_batching_limits_split_rows_by_width(monkeypatch):
    """Width-based limits keep rows*width within token and KV budgets."""
    context, calls = install_stub(monkeypatch, n_ctx=64)
    plan = prepare_prompt(
        {
            "model": "m",
            "state": "shared",
            "questions": {
                key: {"type": "noul", "instructions": key} for key in "abcd"
            },
        }
    )
    # Distinct suffix lengths force sorted, width-limited batches.
    branches = [
        Branch(plan.questions[i].branch_id, list(range(10)) + [20 + i] * (i + 1),
                [1, 2, 3], [], "")
        for i in range(4)
    ]
    compiled = CompiledRequest(plan, branches)
    result = await backend(context, max_batch_size=2, max_batch_tokens=8).score(
        compiled
    )
    sizes = result.metrics["suffix_batch_sizes"]
    assert sum(sizes) == 4
    assert sizes == [2, 2]
    decodes = [call[1] for call in calls if call[0] == "decode"]
    assert len(decodes) == 3  # prefix + two suffix batches


async def test_identical_prompts_leave_a_suffix_token(monkeypatch):
    """Identical prompts share all but the final token, still scoring one row each."""
    context, calls = install_stub(monkeypatch)
    plan = prepare_prompt(
        {
            "model": "m",
            "state": "same",
            "questions": {k: {"type": "noul", "instructions": k} for k in "xy"},
        }
    )
    ids = list(range(1, 8))
    compiled = CompiledRequest(
        plan, [Branch(q.branch_id, ids, [7, 8], [], "") for q in plan.questions]
    )
    result = await backend(context).score(compiled)
    assert result.metrics["prefix_tokens"] == len(ids) - 1
    assert result.metrics["suffix_batch_sizes"] == [2]
    # Two distinct rows mean two flagged reads with per-branch label maps.
    assert set(result.logits) == {"0", "1"}
    assert set(result.logits["0"]) == {"1", "2"} or set(result.logits["0"]) == set(
        plan.questions[0].output_labels
    )


async def test_decode_failure_and_plan_requirements(monkeypatch):
    """Engine failures surface as RuntimeError; scoring requires the plan."""
    context, _ = install_stub(monkeypatch, decode_rc=-1)
    with pytest.raises(RuntimeError, match="llama_decode failed"):
        await backend(context).score(compiled_request())

    context, _ = install_stub(monkeypatch)
    plan = prepare_prompt(
        {
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "?"}},
        }
    )
    compiled = CompiledRequest(
        plan, [Branch("0", [1, 2, 3], [4], [], "")]
    )
    compiled_no_plan = CompiledRequest(
        None, [Branch("0", [1, 2, 3], [4], [], "")]
    )
    assert await backend(context).score(compiled)
    with pytest.raises(ValueError, match="compiled plan"):
        await backend(context).score(compiled_no_plan)


async def test_cancellation_observed_between_batches(monkeypatch):
    """A cancelled score() re-raises after flagging its worker's stop event.

    The worker checks the flag before every batch; with the event set before
    the worker's first check, _score raises CancelledError without any
    decode, proving the cooperative stop runs under the model lock.
    """
    context, calls = install_stub(monkeypatch)
    instance = backend(context)
    compiled = compiled_request()

    # A completed score decodes the prefix and the suffix batches.
    await instance.score(compiled)
    decodes = [call for call in calls if call[0] == "decode"]
    assert len(decodes) >= 2

    # score() forwards cancellation into its worker thread.
    task = asyncio.create_task(instance.score(compiled))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def build_real_service(gguf, **kwargs):
    from hf_server import DecisionService, LlamaCppTokenizer, create_app, load_service

    return load_service(gguf, device=DEVICE, dtype="float32", **kwargs)


# Uneven suffixes exercise padding-free packing of rows with different lengths.
UNEVEN_SUFFIXES = [[100, 101], [102, 103, 104], [105]]
EVEN_SUFFIXES = [[100, 101], [102, 103], [105, 106]]
# Recurrent/hybrid memory slices packed sequences into equal-length sub-batches,
# so a row packed with longer rows has its recurrence computed in several pieces
# instead of one pass. That is float rounding, not a scoring error: measured
# 0.026-0.029 logits on Qwen3.5-0.8B-BF16 (CPU), while a misrouted logits row
# differs by ~10. Equal-length rows are not sliced and must match strictly.
STATEFUL_UNEVEN_TOLERANCE = 0.1


@real_engine
@pytest.mark.parametrize("share_prefix", [True, False], ids=["shared_prefix", "per_branch"])
async def test_shared_prefix_scores_match_independent_decodes(share_prefix):
    """Backend scores equal independent full-prompt decodes on the same context.

    Both prefill strategies the server can select (--prefix-sharing) are
    checked. With SIMPLE_JEV_DEVICE=cpu the context also disables op offload, so
    a GPU-enabled wheel (e.g. Vulkan) cannot move large batches to the GPU and
    CPU remains the numerical reference. Attention-only models must match within
    1e-3 even for uneven packed rows. Recurrent/hybrid models must match within
    1e-3 for equal-length rows and within STATEFUL_UNEVEN_TOLERANCE for uneven
    ones (see above). On other devices, the documented GPU tolerance applies.
    """
    import llama_cpp

    llama_cpp.llama_backend_init()
    model_params = llama_cpp.llama_model_default_params()
    model_params.n_gpu_layers = 0 if DEVICE == "cpu" else -1
    model = _internals.LlamaModel(path_model=GGUF, params=model_params, verbose=False)
    context_params = llama_cpp.llama_context_default_params()
    context_params.n_ctx = 512
    context_params.n_batch = 512
    context_params.n_seq_max = 4
    context_params.kv_unified = True
    if DEVICE == "cpu":
        context_params.op_offload = False
    context = _internals.LlamaContext(
        model=model, params=context_params, verbose=False
    )
    stateful = bool(
        llama_cpp.llama_model_is_recurrent(model.model)
        or llama_cpp.llama_model_is_hybrid(model.model)
    )
    try:
        toks = model.tokenize(
            b"Answer with one word only.", add_bos=True, special=False
        )
        prefix = list(toks[:10])
        # Noul questions score nine labels; give every label its own id.
        output_ids = [10 + i for i in range(9)]
        plan = prepare_prompt(
            {
                "model": "m",
                "state": "s",
                "questions": {
                    key: {"type": "noul", "instructions": key}
                    for key in "abc"
                },
            }
        )

        async def worst_difference(suffixes):
            # Independent references: one fresh sequence per full prompt.
            reference = []
            for suffix in suffixes:
                llama_cpp.llama_memory_clear(context.memory, True)
                full = prefix + suffix
                batch = llama_cpp.llama_batch_init(len(full), 0, 1)
                batch.n_tokens = len(full)
                for j, token_id in enumerate(full):
                    batch.token[j] = token_id
                    batch.pos[j] = j
                    batch.seq_id[j][0] = 0
                    batch.n_seq_id[j] = 1
                    batch.logits[j] = j == len(full) - 1
                assert llama_cpp.llama_decode(context.ctx, batch) == 0
                logits = llama_cpp.llama_get_logits_ith(context.ctx, len(full) - 1)
                reference.append([float(logits[i]) for i in output_ids])
                llama_cpp.llama_batch_free(batch)

            branches = [
                Branch(plan.questions[i].branch_id, prefix + suffixes[i], output_ids,
                       [], "")
                for i in range(3)
            ]
            result = await LlamaCppBackend(model, context, share_prefix=share_prefix).score(
                CompiledRequest(plan, branches)
            )
            assert result.metrics["prefill_strategy"] == (
                "shared_prefix" if share_prefix else "per_branch"
            )
            worst = 0.0
            for i, question in enumerate(plan.questions):
                assert set(result.logits[question.branch_id]) == set(
                    question.output_labels
                ), result.logits
                got = [
                    result.logits[question.branch_id][label]
                    for label in question.output_labels
                ]
                worst = max(worst, max(abs(a - b) for a, b in zip(got, reference[i])))
            return worst

        # CPU is the numerical reference; GPU kernels deviate by up to ~0.1.
        strict = 1e-3 if DEVICE == "cpu" else 0.15
        strategy = "shared_prefix" if share_prefix else "per_branch"
        if stateful:
            worst = await worst_difference(EVEN_SUFFIXES)
            assert worst < strict, f"{strategy} equal-length scores diverged by {worst}"
            worst = await worst_difference(UNEVEN_SUFFIXES)
            limit = max(strict, STATEFUL_UNEVEN_TOLERANCE)
            assert worst < limit, f"{strategy} uneven scores diverged by {worst}"
        else:
            worst = await worst_difference(UNEVEN_SUFFIXES)
            assert worst < strict, f"{strategy} scores diverged by {worst}"
    finally:
        context.close()
        model.close()


@real_engine
async def test_gguf_service_end_to_end_http():
    """Real GGUF service answers all three question types over HTTP."""
    import httpx

    from hf_server import create_app

    service = build_real_service(GGUF, max_model_len=min(1024, 8192))
    payload = {
        "model": GGUF,
        "state": "Mia owns a red bicycle. Her dog is named Max.",
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What color is Mia's bicycle?",
                "criteria": {"red": None, "blue": None},
            },
            "yes": {"type": "noul", "instructions": "Is the dog named Max?"},
            "level": {
                "type": "score",
                "instructions": "Bicycle support?",
                "criteria": ["none", "some"],
            },
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)), base_url="http://t"
    ) as client:
        for route in ["/v1/classifier", "/v1/systemone"]:
            response = await client.post(route, json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["usage"]["output_tokens"] == 0
            assert set(body["answers"]) == {"color", "yes", "level"}
            assert 0.01 <= body["answers"]["yes"]["noul"] <= 0.99
            assert "confidence" in body["answers"]["color"]
    service.backend.context.close()
    service.backend.model.close()

@real_engine
async def test_gguf_discovery_served_name_and_extended_choice_http():
    """Discovery, served names, and >50-option Choice through the real tokenizer."""
    import httpx

    from hf_server import create_app

    service = build_real_service(
        GGUF, max_model_len=8192, served_model_name="public-gguf", prompt_policy="baseline"
    )
    options = {f"item-{i}": None for i in range(64)}
    payload = {
        "model": "any-name",
        "state": "The requested item is item-7.",
        "questions": {
            "item": {"type": "choice", "instructions": "Which item?", "criteria": options},
            "small": {"type": "choice", "instructions": "Pick one.",
                      "criteria": {"a": None, "b": None}},
        },
    }
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)), base_url="http://t"
        ) as client:
            models = (await client.get("/v1/models")).json()["data"]
            assert [(m["id"], m["x_max_choice_options"]) for m in models] == [("public-gguf", 255)]
            response = await client.post("/v1/classifier", json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["model"] == "public-gguf"
            probabilities = body["answers"]["item"]["probabilities"]
            assert list(probabilities) == list(options)
            assert sum(probabilities.values()) == pytest.approx(1, abs=1e-6)
            assert len(body["answers"]["small"]["probabilities"]) == 2
    finally:
        service.backend.context.close()
        service.backend.model.close()
