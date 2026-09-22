"""Single-file GGUF classifier server using llama.cpp (llama-cpp-python).

Run directly from a checkout (the sibling common/ folder is required)::

    python hf-server/hf_server.py --model /path/to/model.gguf --device cpu

Or install with pip install -e './hf-server[test]' and run::

    simple-jev --model organization/model --device auto
    python -m hf_server --model organization/model --device auto

Request flow:
    llama.cpp: ClassifierRequest -> PromptCompiler -> LlamaCppBackend -> common.build_response
    Laya: ClassifierRequest -> LayaBackend -> native encoder scoring -> response

common owns validation, versioned classifier wording, label semantics, and answer
math. This file owns text-only role assembly, native chat/tokenizer boundaries,
shared-prefix inference, queue/cancellation controls, usage accounting, HTTP, and
startup. Model weights load only when load_service/main is called.

The inference engine is llama.cpp through llama-cpp-python. Models must be GGUF
files (or Hugging Face repositories containing one); weights are offloaded to
Vulkan-capable GPUs when --device auto requests it. The engine never samples
tokens: it scores the next-token logits of each branch's answer boundary.

The sections below follow the data flow and retain the implementation notes for
KV ownership, per-row logit selection, and asynchronous cleanup. Tests live in
tests/; no separate simple_jev package or duplicate classifier template is needed.
"""

import argparse
import asyncio
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

# Direct script execution puts hf-server/, not the checkout root, on sys.path.
# Prefer the sibling common source when running from this repo; an installed
# wheel instead imports its bundled common package through normal resolution.
_checkout_root = Path(__file__).resolve().parent.parent
if (_checkout_root / "common" / "prompt_builder.py").is_file():
    sys.path.insert(0, str(_checkout_root))

from common import (
    ClassifierRequest,
    PromptPlan,
    build_response,
    prepare_prompt,
)
from common.prompt_builder import DEFAULT_TEMPLATE_VERSION, canonical

# Shared plan to native chat and tokens


@dataclass(frozen=True)
class Branch:
    """One compiled question, identified by its plan-local branch ID.

    token_ids encodes the complete prompt through the incomplete assistant answer
    prefix. output_ids contains the next-token vocabulary IDs, in the same order
    as the shared question's output_labels. Messages/prefix remain available for
    inspection; they are not reconstructed from tokens during inference.

    render_only compilation leaves both ID lists empty and is not executable.
    frozen prevents attribute reassignment, not mutation of the contained lists.
    """

    branch_id: str
    token_ids: list[int]
    output_ids: list[int]
    messages: list[dict]
    answer_prefix: str


@dataclass
class CompiledRequest:
    """Bind the semantic prompt plan to its HF-specific executable branches.

    Keep this pair together until response scoring: branch IDs and output order
    must be interpreted against the plan that created them. Branch order initially
    matches plan order; the backend may reorder execution for efficient padding.
    """

    plan: PromptPlan
    branches: list[Branch]


def common_prefix(sequences):
    """Return the longest identical token prefix of a nonempty sequence group.

    The caller validates nonempty prompts. An empty shared prefix is valid.
    Comparing actual IDs is essential: matching rendered string fragments alone
    does not prove that the tokenizer produced reusable prefix tokens.
    """
    first = sequences[0]
    end = min(map(len, sequences))
    for other in sequences[1:]:
        for i in range(end):
            if first[i] != other[i]:
                end = i
                break
    return first[:end]


class PromptCompiler:
    """Render shared classifier plans with a model's native tokenizer template."""

    def __init__(self, tokenizer, max_tokens=16384, version=DEFAULT_TEMPLATE_VERSION):
        """Store the renderer, complete-prompt token limit, and shared version.

        Version validation is performed by common.prepare_prompt during compile.
        This class does not load a tokenizer or model on its own.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.version = version

    def compile(self, request: ClassifierRequest, *, render_only=False):
        """Validate text input and compile one branch per shared-plan question.

        request may be a ClassifierRequest or an input dictionary. render_only
        returns roles/content and the shared plan for diagnostics/tests; it skips
        chat-template tokenization, token limits, and output-boundary checks.
        Such a result must not be sent to LlamaCppBackend.score.

        Unsupported media/tools, malformed token boundaries, and overlong prompts
        raise ValueError. Caller data is preserved when merging system messages.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        # The shared schema allows richer contexts for other integrations. This
        # adapter narrows that contract before constructing text-only messages.
        if request.tools or request.mm_processor_kwargs or request.media_io_kwargs:
            raise ValueError(
                "HF text reference does not support tools or media options"
            )
        if request.messages and any(
            not isinstance(m.content, str)
            or m.model_extra
            or m.role in {"tool", "function"}
            for m in request.messages
        ):
            raise ValueError("HF text reference accepts plain text chat only")

        plan = prepare_prompt(request, version=self.version)
        system = plan.system_prompt_prefix + plan.prefix_instruction
        branches = []
        for question in plan.questions:
            content = plan.suffix_instruction + question.instruction
            # State is JSON-serialized, including quotes around string states.
            # For chat, preserve turn boundaries and merge only a leading system
            # turn; the final selected question is always a new user message.
            if request.messages is None:
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"State:\n{canonical(request.state)}\n\n" + content,
                    },
                ]
            else:
                # Dump into new dictionaries so adding classifier instructions
                # never mutates the caller's existing conversation.
                messages = [m.model_dump(exclude_none=True) for m in request.messages]
                if messages[0]["role"] == "system":
                    messages[0]["content"] = system + "\n" + messages[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": system})
                messages.append({"role": "user", "content": content})

            ids, output_ids = [], []
            if not render_only:
                # Render first, then append incomplete JSON to the open assistant
                # position. Do not create a completed assistant message or add
                # a closing brace/EOS before the next-token scoring position.
                text = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    + question.answer_prefix
                )
                # The template already supplies special tokens. Adding another
                # BOS/EOS during encode would alter the intended model input.
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not ids or len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Branch for {question.question_id!r} must contain 1–{self.max_tokens} tokens"
                    )
                # Derive IDs at the actual rendered boundary, not from isolated
                # label encoding. Check every branch; templates/context can affect it.
                for label in question.output_labels:
                    extended = self.tokenizer.encode(
                        text + label, add_special_tokens=False
                    )
                    if len(extended) != len(ids) + 1 or extended[:-1] != ids:
                        raise ValueError(
                            f"Answer label {label!r} is not single-token stable"
                        )
                    output_ids.append(extended[-1])
                if len(set(output_ids)) != len(output_ids):
                    raise ValueError("Output labels must map to distinct token IDs")
            branches.append(
                Branch(
                    question.branch_id,
                    ids,
                    output_ids,
                    messages,
                    question.answer_prefix,
                )
            )
        return CompiledRequest(plan, branches)


# Inference result


@dataclass
class BackendResult:
    """Results keyed by plan-local branch ID, independent of batch execution order.

    This server stores one label-to-float mapping per branch, keyed by the
    shared plan's exact output labels and ordered like the corresponding
    question's output_labels. The broad Any annotation also permits the same
    form from test backends, accepted by common's scorer.

    metrics contains internal timing/token/batch counters. branch_output_tokens
    is zero for logits-only inference. It does not supply input usage; the
    service computes the unique token-prefix union from the compiled prompts.
    """

    # Compact tensors in each plan question's output_labels order, keyed by ID.
    logits: dict[str, Any]
    metrics: dict


# Logical input-token accounting


def unique_prompt_tokens(sequences):
    """Return the number of distinct token-tree edges across all sequences.

    Example: [1, 2, 3] and [1, 2, 4] count as four tokens, not six. Duplicate
    paths add nothing; a path that ends inside another adds no new suffix. Empty
    input returns zero. Sorting creates a new list and leaves caller order intact.

    Lexicographic neighbors share the greatest already-counted prefix for each
    next sequence. Subtract that overlap from its length instead of building an
    explicit trie. No global cache or backend warmup information is consulted.
    """
    # Lexicographically adjacent sequences share the maximum previously seen
    # prefix. Count each token-tree edge once, including hierarchical prefixes.
    total = 0
    previous = []
    for sequence in sorted(sequences):
        shared = 0
        for left, right in zip(previous, sequence):
            if left != right:
                break
            shared += 1
        total += len(sequence) - shared
        previous = sequence
    return total


# Shared-prefix model execution


class LlamaCppBackend:
    """Own one llama.cpp context and serialize all decodes through a thread lock.

    The context is a unified KV pool of n_ctx cells shared by sequence ids:
    sequence 0 holds the request's common prefix; each suffix-batch row runs on
    its own sequence id 1..N after a llama_memory_seq_cp copies the prefix cells.
    Rows are packed into one llama_decode call per batch (no padding tokens),
    with logits requested only at each row's final suffix position.
    """

    def __init__(
        self, model, context, *, max_batch_size=32, max_batch_tokens=32768,
        share_prefix=True,
    ):
        """Bind one loaded model/context and bounds on packed suffix batches.

        max_batch_size caps rows per decode and sequence ids (1 + rows must fit
        the context's n_seq_max). max_batch_tokens caps the packed suffix token
        count per decode; it does not bound the prefix prefill. The compiler
        enforces the full model-context limit separately.
        """
        if max_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch limits must be positive")
        self.model = model
        self.context = context
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        # False makes every branch prefill its own full prompt instead of
        # copying shared prefix cells, the only strategy that is valid when
        # llama.cpp cannot duplicate an architecture's rolling state.
        self.share_prefix = share_prefix
        # Protect the context even if cancellation returns before its thread exits.
        self._lock = threading.Lock()

    async def score(self, compiled):
        """Run blocking engine work in a worker thread with cooperative cancellation.

        Cancelling an asyncio task cannot interrupt an in-flight GPU kernel.
        Set a stop flag for the worker to check before its next decode, then let
        cancellation propagate. The context lock remains held until the worker exits.
        """
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._score, compiled, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _decode_rows(self, llama_cpp, rows):
        """Pack rows into one llama_batch and decode; return flagged positions.

        rows are (seq_id, pos_start, token_ids, want_last_logits) tuples. Every
        field of every used slot is explicitly initialized: llama_batch memory
        is recycled by the allocator and may otherwise contain garbage. Batch
        arrays are written through one cached handle each to avoid re-fetching
        the ctypes pointers for every element.
        """
        n_tokens = sum(len(tokens) for _, _, tokens, _ in rows)
        batch = llama_cpp.llama_batch_init(n_tokens, 0, 1)
        try:
            batch.n_tokens = n_tokens
            token, pos = batch.token, batch.pos
            seq_id, n_seq_id, logits = batch.seq_id, batch.n_seq_id, batch.logits
            index = 0
            flagged = []
            for seq, start, tokens, want in rows:
                last = len(tokens) - 1
                for offset, token_id in enumerate(tokens):
                    token[index] = int(token_id)
                    pos[index] = start + offset
                    seq_id[index][0] = int(seq)
                    n_seq_id[index] = 1
                    logits[index] = bool(want and offset == last)
                    index += 1
                if want:
                    flagged.append(index - 1)
            if llama_cpp.llama_decode(self.context.ctx, batch) != 0:
                raise RuntimeError("llama_decode failed for the scoring batch")
            return flagged
        finally:
            llama_cpp.llama_batch_free(batch)

    def _score(self, compiled, stop):
        """Execute one request; all model/KV work stays under the same lock."""
        import llama_cpp

        with self._lock:
            if stop.is_set():
                raise asyncio.CancelledError()
            start = time.perf_counter()
            sequences = [b.token_ids for b in compiled.branches]
            if not sequences or any(not ids for ids in sequences):
                raise ValueError("Expected nonempty scoring prompts")
            if compiled.plan is None:
                raise ValueError("Backend scoring requires the compiled plan")
            labels = {
                question.branch_id: question.output_labels
                for question in compiled.plan.questions
            }
            # Leave at least one suffix token, including for identical prompts.
            prefix = (
                common_prefix(sequences)[: min(map(len, sequences)) - 1]
                if self.share_prefix
                else []
            )
            memory = self.context.memory
            # Start every request from an empty KV pool: prefix work never
            # persists across requests, so results cannot leak between callers.
            llama_cpp.llama_memory_clear(memory, True)
            decodes = 0
            if prefix:
                # One unchunked prefill decode on sequence 0. No logits are
                # needed here; every scoring position lives in the suffixes.
                self._decode_rows(llama_cpp, [(0, 0, prefix, False)])
                decodes += 1
            lengths = [len(ids) - len(prefix) for ids in sequences]
            if max(lengths) > self.max_batch_tokens:
                raise ValueError("A question suffix exceeds max_batch_tokens")
            # Longest first groups equal lengths together. Results carry branch
            # IDs, so execution order need not equal the original question order.
            pending = sorted(
                range(len(sequences)), key=lambda i: lengths[i], reverse=True
            )
            results = {}
            sizes = []
            packed_tokens = 0
            n_ctx = self.context.n_ctx()
            while pending:
                if stop.is_set():
                    raise asyncio.CancelledError()
                # Width-based limits mirror the previous engine's padded-batch
                # accounting: rows times width bounded by max_batch_tokens, and
                # every live KV cell (prefix plus suffixes) bounded by n_ctx.
                width = lengths[pending[0]]
                limit = min(
                    self.max_batch_size,
                    self.max_batch_tokens // width if width else 1,
                    max(1, (n_ctx - len(prefix)) // width) if width else 1,
                )
                batch, pending = pending[:limit], pending[limit:]
                # Each row runs on its own sequence id sharing the prefix cells:
                # the unified cache marks copied cells for both sequences, so
                # attention sees prefix+suffix without recomputing the prefill.
                if prefix:
                    for row in range(len(batch)):
                        llama_cpp.llama_memory_seq_cp(
                            memory, 0, row + 1, 0, len(prefix)
                        )
                rows = [
                    (row + 1, len(prefix), sequences[i][len(prefix):], True)
                    for row, i in enumerate(batch)
                ]
                flagged = self._decode_rows(llama_cpp, rows)
                # One flagged output per row, in batch order: read exactly the
                # permitted label logits and copy them to plain Python floats.
                for row, i in enumerate(batch):
                    branch = compiled.branches[i]
                    logits = llama_cpp.llama_get_logits_ith(
                        self.context.ctx, flagged[row]
                    )
                    results[branch.branch_id] = {
                        label: float(logits[token_id])
                        for label, token_id in zip(
                            labels[branch.branch_id], branch.output_ids
                        )
                    }
                # Row sequences are discarded after each batch: seq_rm drops
                # their claim on the shared prefix cells and frees their suffix
                # cells, making both the cells and the sequence ids reusable.
                for row in range(len(batch)):
                    llama_cpp.llama_memory_seq_rm(memory, row + 1, 0, -1)
                decodes += 1
                sizes.append(len(batch))
                packed_tokens += sum(lengths[i] for i in batch)
            return BackendResult(
                results,
                {
                    "backend": "llama-cpp",
                    "prefill_strategy": (
                        "shared_prefix" if self.share_prefix else "per_branch"
                    ),
                    "prefix_tokens": len(prefix),
                    "suffix_batch_sizes": sizes,
                    "engine_forwards": decodes,
                    # These are distinct accounting views, not interchangeable:
                    # branch_prompt_tokens repeats shared context per branch;
                    # computed_prompt_tokens counts cells the engine filled;
                    # logical_prefill_tokens counts each branch's suffix once
                    # beyond the single shared prefix. Rows are packed without
                    # padding tokens, so the latter two views agree here.
                    "branch_prompt_tokens": sum(map(len, sequences)),
                    "computed_prompt_tokens": len(prefix) + packed_tokens,
                    "logical_prefill_tokens": len(prefix) + sum(lengths),
                    "padded_suffix_tokens": packed_tokens,
                    "branch_output_tokens": 0,
                    "scored_positions": len(sequences),
                    "backend_seconds": time.perf_counter() - start,
                },
            )


# Request admission and response assembly


def validate_rope_factor(factor):
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("RoPE factor must be finite and at least 1")


def configure_rope(context_params, factor):
    """Configure llama.cpp linear position interpolation on the context.

    llama.cpp applies rope_freq_scale = 1/factor as its linear rope scaling:
    position p then uses the original rotary angle at p/factor, matching the
    engine's --rope-scale behavior. The admission limit stays governed by
    --max-model-len; extending positional capacity does not establish accuracy,
    and interpolation also changes short-input behavior.
    """
    validate_rope_factor(factor)
    if factor == 1:
        return
    context_params.rope_scaling_type = 1  # LLAMA_ROPE_SCALING_TYPE_LINEAR
    context_params.rope_freq_scale = 1.0 / float(factor)


def extend_laya_rope(agent, factor):
    """Experimental linear position interpolation for Laya's ModernBERT encoder.

    Dividing both full/sliding inverse frequencies by two maps position p to
    its original rotary angle at p/2. The local attention window is unchanged.
    This changes short-input behavior too; it does not establish longer-context
    accuracy. Refuse other architectures or already-scaled RoPE rather than
    silently composing incompatible scaling rules.
    """
    if factor == 1:
        return
    validate_rope_factor(factor)
    encoder = agent.model.encoder
    rotary = getattr(encoder, "rotary_emb", None)
    if encoder.config.model_type != "modernbert" or rotary is None:
        raise ValueError("Laya RoPE extension currently requires ModernBERT")
    if getattr(agent, "_simple_jev_rope_extended", False):
        raise ValueError("Laya RoPE has already been extended")
    kinds = ("full_attention", "sliding_attention")
    for kind in kinds:
        if rotary.rope_type.get(kind) != "default" or not hasattr(
            rotary, f"{kind}_inv_freq"
        ):
            raise ValueError("Laya RoPE extension requires unscaled full/sliding RoPE")
    target = int(int(agent.cfg["max_len"]) * factor)
    if target > encoder.config.max_position_embeddings:
        raise ValueError("Extended Laya sequence exceeds encoder position capacity")
    import torch

    with torch.no_grad():
        for kind in kinds:
            getattr(rotary, f"{kind}_inv_freq").div_(factor)
            getattr(rotary, f"{kind}_original_inv_freq").div_(factor)
    agent.cfg["max_len"] = target
    agent._simple_jev_rope_extended = True


class LayaBackend:
    """Native encoder adapter; no chat prefill, vocabulary labels, or KV cache.

    The SDK owns option-marker formatting and temperature scaling. We retain its
    binary Noul probability, normalize confidence to Simple Jev's max probability,
    and omit SDK-only action fields. A lock protects the model even if the HTTP
    coroutine is cancelled while its worker is finishing.
    """

    def __init__(self, agent, max_tokens):
        self.agent = agent
        self.max_tokens = min(max_tokens, int(agent.cfg["max_len"]))
        self._lock = threading.Lock()

    async def classify_native(self, request):
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._classify, request, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _classify(self, request, stop):
        import math
        from laya.common import build_sequence, serialize_state

        with self._lock:
            if stop.is_set():
                raise asyncio.CancelledError()
            if request.tools or request.mm_processor_kwargs:
                raise ValueError("Laya supports text state and text chat only")
            if request.options.raw_logits:
                raise ValueError("Laya raw_logits diagnostics are not supported")
            state = request.state
            if request.messages is not None:
                # Chat is serialized as role/content data, not a causal chat template.
                turns = []
                for message in request.messages:
                    if (
                        not isinstance(message.content, str)
                        or message.role not in {"system", "user", "assistant"}
                        or message.model_extra
                    ):
                        raise ValueError(
                            "Laya supports plain system/user/assistant text messages only"
                        )
                    turns.append({"role": message.role, "content": message.content})
                state = turns
            questions = {key: q.model_dump() for key, q in request.questions.items()}
            # SDK truncates state by default. Probe with a generous token budget
            # and reject overflow so important context cannot disappear silently.
            for key, definition in questions.items():
                q = self.agent._to_internal(definition)
                state_size = len(
                    self.agent.tok(
                        serialize_state(state).replace(self.agent.tok.mask_token, " "),
                        add_special_tokens=False,
                    )["input_ids"]
                )
                ids, _ = build_sequence(
                    self.agent.tok,
                    state,
                    q,
                    state_size + self.agent.cfg.get("head_max_len", 192) + 65536,
                    self.agent.cfg.get("head_max_len", 192),
                )
                if len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Laya question {key!r} exceeds {self.max_tokens} input tokens"
                    )
            if stop.is_set():
                raise asyncio.CancelledError()
            result = self.agent.predict(state, questions)
            if set(result["answers"]) != set(questions):
                raise ValueError("Laya returned incomplete answers")
            answers = {}
            for key, q in questions.items():
                source = result["answers"][key]
                if q["type"] == "noul":
                    value = float(source["noul"])
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Laya returned an invalid probability")
                    answers[key] = {"type": "noul", "noul": value}
                    continue
                labels = (
                    list(q["criteria"])
                    if q["type"] == "choice"
                    else [str(i) for i in range(len(q["criteria"]))]
                )
                probs = source["probabilities"]
                if (
                    set(probs) != set(labels)
                    or any(
                        not math.isfinite(float(v)) or not 0 <= float(v) <= 1
                        for v in probs.values()
                    )
                    or abs(sum(probs.values()) - 1) > 0.01
                ):
                    raise ValueError("Laya returned invalid option probabilities")
                answer = {
                    "type": q["type"],
                    "probabilities": probs,
                    "confidence": max(probs.values()),
                }
                if q["type"] == "choice":
                    if source["choice"] not in labels:
                        raise ValueError("Laya returned an invalid choice")
                    answer["choice"] = source["choice"]
                else:
                    value = float(source["score"])
                    if not math.isfinite(value) or not 0 <= value <= len(labels) - 1:
                        raise ValueError("Laya returned an invalid score")
                    answer.update(
                        score=value,
                        legend={str(i): c for i, c in enumerate(q["criteria"])},
                    )
                answers[key] = answer
            return {
                "model": request.model,
                "answers": answers,
                "usage": result["usage"],
            }


class OverloadedError(Exception):
    """Admission capacity is exhausted; the HTTP layer translates this to 429."""


class DecisionService:
    """Manage one configured model, its compiler/backend, and request lifecycle.

    Use on one asyncio event loop: the admission counter is intentionally updated
    without awaiting between its capacity check and increment. Compiler/backend
    dependencies allow service tests to run without model weights.
    """

    def __init__(
        self,
        model,
        compiler,
        backend,
        *,
        metadata=None,
        concurrency=4,
        queue_size=16,
        max_request_branches=100,
        model_aliases=(),
        advanced_metrics=None,
    ):
        """Configure admission and diagnostic output for a loaded model.

        concurrency counts active coroutine slots; queue_size adds waiting slots.
        Supply a positive concurrency and nonnegative queue size. model_aliases
        permits additional names for the same loaded model, not dynamic loading.
        advanced_metrics explicitly overrides the environment flag when provided;
        otherwise 1/true/yes/on enable ENABLE_OPEN_JEV_ADVANCED_METRICS.
        """
        if max_request_branches < 1:
            raise ValueError("max_request_branches must be positive")
        self.max_request_branches = max_request_branches
        self.model_aliases = {model, *model_aliases}
        self.model = model
        self.compiler = compiler
        self.backend = backend
        self.metadata = metadata or {}
        self.advanced_metrics = (
            os.environ.get("ENABLE_OPEN_JEV_ADVANCED_METRICS", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if advanced_metrics is None
            else advanced_metrics
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._capacity = concurrency + queue_size
        self._inflight = 0

    async def classify(self, request):
        """Return a complete response or raise validation/overload/cancellation.

        Validate and admit before expensive work. Admission is released in finally
        on success, failure, or cancellation. The queue timer ends when the active
        slot is acquired; total_seconds spans admitted work through response build.
        These timings use a monotonic clock and are not distributed trace spans.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        if request.model not in self.model_aliases:
            raise ValueError(f"Loaded model is {self.model!r}")
        # v1 has exactly one inference branch per question; candidate count no
        # longer expands requests. The shared schema separately caps 256 questions.
        branches = len(request.questions)
        if branches > self.max_request_branches:
            raise ValueError(
                f"Request has {branches} scoring branches; maximum is {self.max_request_branches}"
            )
        if self._inflight >= self._capacity:
            raise OverloadedError("Scoring queue is full")
        # No await between this check/increment pair: other tasks on the same
        # loop cannot interleave admission and oversubscribe the capacity.
        self._inflight += 1
        start = time.perf_counter()
        try:
            async with self._semaphore:
                queued = time.perf_counter() - start
                if hasattr(self.backend, "classify_native"):
                    response = await self.backend.classify_native(request)
                    if self.advanced_metrics:
                        response["metadata"] = {
                            **self.metadata,
                            "format": "laya-native",
                            "usage_accounting": "sum_of_question_sequence_tokens",
                        }
                        response["metrics"] = {
                            "queue_seconds": queued,
                            "total_seconds": time.perf_counter() - start,
                        }
                    return response
                # Tokenization can be expensive and must not block cancellation/HTTP.
                if hasattr(self.compiler, "compile_async"):
                    compiled = await self.compiler.compile_async(request)
                else:
                    if request.messages and any(
                        not isinstance(m.content, str) or m.role in {"tool", "function"}
                        for m in request.messages
                    ):
                        raise ValueError(
                            "Multimodal/tool chat requires a native renderer; this backend accepts text chat only"
                        )
                    compiled = await asyncio.to_thread(self.compiler.compile, request)
                # The compiler retains the shared plan alongside executable IDs.
                # Never reconstruct label meaning from model output or batch order.
                result = await self.backend.score(compiled)
                response = build_response(
                    compiled.plan,
                    result.logits,
                    # Logical prefix-union accounting excludes batch padding and
                    # repeated context, not a sum of the backend's forward sizes.
                    input_tokens=unique_prompt_tokens(
                        [b.token_ids for b in compiled.branches]
                    ),
                    output_tokens=result.metrics.get("branch_output_tokens", 0),
                    advanced=self.advanced_metrics,
                )
                # Common handles answer filtering and authoritative version data;
                # the server only adds backend-specific metadata and timings.
                if self.advanced_metrics:
                    response["metadata"] = {
                        **self.metadata,
                        **response["metadata"],
                        "usage_accounting": "unique_token_prefixes_and_engine_leaf_outputs",
                    }
                    response["metrics"] = {
                        **result.metrics,
                        "queue_seconds": queued,
                        "total_seconds": time.perf_counter() - start,
                    }
                return response
        finally:
            # Release service admission even if a cancelled worker is finishing.
            # The backend's lock still prevents overlapping access to the model.
            self._inflight -= 1


# HTTP routes and errors


def validation_response(message=None, errors=()):
    """Format explicit text or Pydantic errors without returning input payloads.

    Keep at most ten structured details and report the count of additional
    errors. Convert location tuples into readable dotted paths with array indices,
    omitting the leading transport-specific 'body' component. The first location
    becomes the top-level param; errors without a location use null.
    """
    details = []
    for error in errors[:10]:
        path = ""
        for part in error.get("loc", ()):
            if part == "body" and not path:
                continue
            if isinstance(part, int):
                path += f"[{part}]"
            else:
                path += ("." if path else "") + str(part)
        details.append(
            {"param": path or None, "message": error["msg"], "type": error["type"]}
        )
    if message is None:
        message = (
            "; ".join(
                f"{e['param']}: {e['message']}" if e["param"] else e["message"]
                for e in details
            )
            or "Invalid classifier request"
        )
        if len(errors) > len(details):
            message += f"; {len(errors) - len(details)} additional validation errors"
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": 422,
                "param": details[0]["param"] if details else None,
                "details": details,
            }
        },
    )


class ClassifierRoute(APIRoute):
    """Normalize errors only for classifier routes, leaving host handlers alone.

    Request parsing can fail before the endpoint function runs, so normalization
    belongs around FastAPI's generated route handler as well as in the endpoint.
    """

    def get_route_handler(self):
        """Wrap FastAPI parsing/execution while preserving non-422 exceptions."""
        handler = super().get_route_handler()

        async def validated(request):
            """Intercept classifier validation errors before they leave this route."""
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Exclude input values and exception contexts from public errors.
                return validation_response(errors=exc.errors())
            except HTTPException as exc:
                if exc.status_code == 422:
                    return validation_response(message=str(exc.detail))
                raise

        return validated


def create_app(service):
    """Build a standalone app around an already-created service.

    The CLI loads the model before calling this function. Readiness therefore
    reports that configured service, not a new inference probe on every request.
    """
    app = FastAPI(title="Simple-JEV", version="0.1.0")

    @app.get("/health")
    async def health():
        """Return the configured model identifier without invoking inference."""
        return {"status": "ready", "model": service.model}

    attach_routes(app, lambda request: service)
    return app


def attach_routes(app, get_service):
    """Attach classifier endpoints; the alias stays out of generated OpenAPI.

    get_service is synchronous and request-scoped, allowing an embedding app to
    select its service without changing the classifier handler's implementation.
    """
    router = APIRouter(route_class=ClassifierRoute)

    @router.post("/v1/classifier")
    @router.post("/v1/systemone", include_in_schema=False)
    async def classify(body: ClassifierRequest, request: Request):
        """Run one service task and cancel it if the HTTP client disconnects."""
        task = asyncio.create_task(get_service(request).classify(body))
        try:
            # Wait with a timeout rather than awaiting task directly, so a long
            # tokenization/model pass does not prevent disconnect checks.
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "Client disconnected")
            return await task
        except OverloadedError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ValidationError as exc:
            return validation_response(errors=exc.errors())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            # Always observe the child task's completion/exception. Cancelling
            # its coroutine does not forcibly stop an active model worker thread;
            # the backend implements cooperative stopping and a model lock.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app.include_router(router)


# Model loading and command-line entry point


def resolve_gguf_path(model_name, revision, gguf_file=None):
    """Return one local GGUF file path for a path, directory, or HF repository.

    A directory must contain exactly one .gguf file (sharded files following
    llama.cpp's <name>-00001-of-000NN.gguf scheme load together and count as
    one). gguf_file selects one file from a directory or Hugging Face repository.
    A Hugging Face repository is downloaded with huggingface_hub, keeping the
    repository identifier as the request-visible model name.
    """
    path = Path(model_name)
    if path.is_file():
        if gguf_file:
            raise ValueError("--gguf-file cannot be used with a GGUF file path")
        return str(path)
    if path.is_dir():
        matches = sorted(path.glob(gguf_file or "*.gguf"))
    else:
        if gguf_file:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Downloading GGUF weights requires huggingface-hub; install it "
                    "or point --model at a local .gguf file"
                ) from exc
            if Path(gguf_file).name != gguf_file or not gguf_file.endswith(".gguf"):
                raise ValueError("--gguf-file must name one .gguf file")
            return hf_hub_download(
                repo_id=model_name, filename=gguf_file, revision=revision
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Downloading GGUF weights requires huggingface-hub; install it "
                "or point --model at a local .gguf file"
            ) from exc
        root = Path(snapshot_download(model_name, revision=revision, allow_patterns=["*.gguf"]))
        matches = sorted(root.glob("*.gguf"))
        # Shards named <name>-00001-of-000NN.gguf are loaded as one model from
        # their first file; report the shard set, not each file, as the choice.
        shards = [m for m in matches if "-of-" in m.name]
        if shards:
            first = shards[0]
            matches = [first]
    if not matches:
        raise ValueError(f"No .gguf file found for model {model_name!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple .gguf files found for {model_name!r}; pass one file path"
        )
    return str(matches[0])


def resolve_n_gpu_layers(device, n_gpu_layers):
    """Map the CLI device selection to llama.cpp layer offloading.

    None means "derive from device": cpu disables offload entirely, while the
    accelerator spellings (auto/gpu/vulkan/cuda/rocm/metal) offload every layer
    so llama.cpp routes through any compiled backend, including Vulkan. An
    explicit n_gpu_layers value always wins over --device.
    """
    if n_gpu_layers is not None:
        if n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be -1 or nonnegative")
        return n_gpu_layers
    normalized = device.strip().lower()
    if normalized == "cpu":
        return 0
    if normalized in {"auto", "gpu", "vulkan", "cuda", "rocm", "metal"}:
        return -1
    raise ValueError(
        f"Unknown device {device!r}; use cpu, auto, gpu, or an explicit --n-gpu-layers"
    )


KV_CACHE_TYPES = {"float32": 0, "float16": 1, "bfloat16": 30}
# ggml_type enum: F32=0, F16=1, BF16=30 in this llama.cpp release. GGUF weights
# keep their own quantization; --dtype selects only the KV cache precision.


class LlamaCppTokenizer:
    """Adapter from the HF-style tokenizer surface to one llama.cpp GGUF vocab.

    PromptCompiler needs exactly two calls: apply_chat_template(tokenize=False)
    and encode(text, add_special_tokens=False). Both are served natively by
    llama.cpp — its tokenizer reads the GGUF's embedded vocabulary, and the
    chat template is rendered from the GGUF's tokenizer.chat_template metadata
    with the same Jinja environment settings llama-cpp-python uses (matching
    Transformers' trim_blocks/lstrip_blocks conventions).
    """

    def __init__(self, model, metadata=None):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._model = model
        metadata = metadata if metadata is not None else model.metadata()
        template = metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(
                "GGUF file has no tokenizer.chat_template metadata; the shared "
                "v1 template requires a chat-formatted model"
            )
        bos_text, eos_text = "", ""
        try:
            bos_id = int(metadata.get("tokenizer.ggml.bos_token_id", ""))
            eos_id = int(metadata.get("tokenizer.ggml.eos_token_id", ""))
            bos_text = model.token_get_text(bos_id) if 0 <= bos_id else ""
            eos_text = model.token_get_text(eos_id) if 0 <= eos_id else ""
        except ValueError:
            pass
        self._template = template
        self._bos_text = bos_text
        self._eos_text = eos_text
        self._formatters = {}

    def _formatter(self, add_generation_prompt):
        """Build (and cache) a Jinja formatter per generation-prompt setting.

        The formatter renders with its own constructor flag, so one cached
        instance exists for each add_generation_prompt value the compiler uses.
        """
        if add_generation_prompt not in self._formatters:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            self._formatters[add_generation_prompt] = Jinja2ChatFormatter(
                template=self._template,
                eos_token=self._eos_text,
                bos_token=self._bos_text,
                add_generation_prompt=add_generation_prompt,
            )
        return self._formatters[add_generation_prompt]

    def encode(self, text, add_special_tokens=False):
        """Tokenize text with llama.cpp's native GGUF tokenizer.

        add_special_tokens=False keeps the template responsible for BOS/EOS:
        the rendered chat text already carries its special tokens, and adding
        another BOS would alter the intended model input.
        """
        return self._model.tokenize(
            text.encode("utf-8"), add_bos=bool(add_special_tokens), special=True
        )

    def apply_chat_template(self, messages, **kwargs):
        """Render the GGUF chat template exactly as the compiler requests it.

        tokenize must stay False: the compiler tokenizes the rendered text.
        add_generation_prompt opens the assistant turn; enable_thinking and
        any other Transformers-style flags are forwarded into the Jinja render,
        the same forwarding Transformers performs for its thinking-capable
        models.
        """
        if kwargs.get("tokenize"):
            raise ValueError("This adapter renders chat templates as text only")
        forward = {
            key: value
            for key, value in kwargs.items()
            if key not in {"tokenize", "add_generation_prompt"}
        }
        formatter = self._formatter(bool(kwargs.get("add_generation_prompt")))
        return formatter(messages=messages, **forward).prompt


def load_service(
    model_name,
    *,
    revision=None,
    gguf_file=None,
    backend="llama-cpp",
    subfolder=None,
    rope_factor=1,
    device="auto",
    dtype="bfloat16",
    n_gpu_layers=None,
    max_model_len=16384,
    max_batch_size=32,
    max_batch_tokens=32768,
    max_request_branches=100,
    prefix_sharing="auto",
):
    """Load a model and return a ready-to-use service, without starting HTTP.

    device selects weight placement: cpu keeps everything on the host, auto
    offloads every layer to any available llama.cpp backend (Vulkan included)
    and falls back to CPU where no device exists. dtype selects the KV cache
    element type. max_model_len limits each complete compiled prompt and sizes
    the unified KV pool. max_batch_tokens limits packed suffix tokens per
    decode, not the shared-prefix prefill or total KV memory.
    max_request_branches caps questions admitted in a single request.

    The loader sets service concurrency to one: separate requests are
    serialized, while branches within a request are batched. The backend's
    thread lock also prevents overlap if cancellation releases admission
    before a decode ends.
    """
    validate_rope_factor(rope_factor)
    if backend == "laya":
        # Resolve the revision ourselves because the SDK does not expose it.
        # Import only when selected; a llama.cpp installation stays usable.
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "Install Laya support with pip install -e './hf-server[laya]'"
            ) from exc
        path = model_name
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                model_name,
                revision=revision,
                allow_patterns=[f"{subfolder}/*"]
                if subfolder
                else [
                    "rl_agent_config.json",
                    "model.safetensors",
                    "encoder/*",
                    "tokenizer/*",
                ],
            )
        agent = laya.load(
            path, subfolder=subfolder, device=None if device == "auto" else device
        )
        extend_laya_rope(agent, rope_factor)
        return DecisionService(
            model_name,
            None,
            LayaBackend(agent, max_model_len),
            concurrency=1,
            max_request_branches=max_request_branches,
            metadata={
                "backend": "laya",
                "model_revision": revision,
                "subfolder": subfolder,
                "rope_factor": rope_factor,
                "native_sequence_limit": agent.cfg["max_len"],
            },
        )
    if backend != "llama-cpp":
        raise ValueError(f"Unknown backend: {backend}")
    if subfolder:
        raise ValueError("--subfolder is currently supported only with --backend laya")
    if dtype not in KV_CACHE_TYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    # Heavy dependencies are local to loading, so CLI help and source inspection
    # do not initialize a backend or import the model classes.
    import llama_cpp
    from llama_cpp import _internals

    gguf_path = resolve_gguf_path(model_name, revision, gguf_file)
    model_params = llama_cpp.llama_model_default_params()
    resolved_layers = resolve_n_gpu_layers(device, n_gpu_layers)
    # llama.cpp encodes "all layers" as INT32 max; -1 is this API's spelling.
    model_params.n_gpu_layers = (
        0x7FFFFFFF if resolved_layers == -1 else resolved_layers
    )
    model = _internals.LlamaModel(
        path_model=gguf_path, params=model_params, verbose=False
    )
    # Recurrent and hybrid architectures carry rolling state alongside (or
    # instead of) attention cells. Recent llama.cpp does duplicate that state in
    # llama_memory_seq_cp, but support varies by architecture and build, so these
    # models default to per-branch prefill: correct everywhere, and only as
    # expensive as the prefill it stops sharing. --prefix-sharing opts back in.
    stateful = bool(
        llama_cpp.llama_model_is_recurrent(model.model)
        or llama_cpp.llama_model_is_hybrid(model.model)
    )
    if prefix_sharing not in ("auto", "on", "off"):
        model.close()
        raise ValueError(f"Unknown prefix sharing mode: {prefix_sharing}")
    share_prefix = not stateful if prefix_sharing == "auto" else prefix_sharing == "on"
    n_ctx_train = model.n_ctx_train()
    if max_model_len > n_ctx_train:
        # The engine clamps its own n_ctx to the trained context; admission and
        # the compiler limit must agree with it or requests would be rejected
        # far from their validation point.
        max_model_len = n_ctx_train
    context_params = llama_cpp.llama_context_default_params()
    context_params.n_ctx = max_model_len
    # One decode must be able to hold the shared prefix and a full token budget.
    # llama.cpp additionally clamps n_batch to n_ctx on its own.
    context_params.n_batch = max(max_batch_tokens, max_model_len)
    context_params.n_seq_max = max_batch_size + 1
    # The unified KV pool is one shared array of n_ctx cells; sequence copies
    # with a partial position range require it (the split cache asserts).
    context_params.kv_unified = True
    context_params.type_k = KV_CACHE_TYPES[dtype]
    context_params.type_v = KV_CACHE_TYPES[dtype]
    configure_rope(context_params, rope_factor)
    context = _internals.LlamaContext(
        model=model, params=context_params, verbose=False
    )
    # PromptCompiler's default comes from common.DEFAULT_TEMPLATE_VERSION.
    # Keep one compiler/backend pair for the service's loaded model/context.
    compiler = PromptCompiler(
        LlamaCppTokenizer(model), max_tokens=max_model_len
    )
    backend = LlamaCppBackend(
        model,
        context,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        share_prefix=share_prefix,
    )
    return DecisionService(
        model_name,
        compiler,
        backend,
        concurrency=1,
        max_request_branches=max_request_branches,
        metadata={
            "backend": "llama-cpp",
            "model_revision": revision,
            "gguf_path": gguf_path,
            "n_gpu_layers": resolved_layers,
            "kv_cache_dtype": dtype,
            "stateful_architecture": stateful,
            "prefix_sharing": share_prefix,
            "rope_factor": rope_factor,
        },
    )


def main():
    """Parse process settings, load the service, then run its ASGI application.

    host/port are Uvicorn settings; the other arguments configure loading and
    inference. Model loading happens before the listener starts accepting work.
    --help exits during parsing and therefore does not load any weights.
    """
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--gguf-file",
        help="GGUF filename to download from a Hugging Face repository",
    )
    parser.add_argument(
        "--backend", choices=["llama-cpp", "laya"], default="llama-cpp"
    )
    parser.add_argument(
        "--subfolder", help="Laya checkpoint subfolder, e.g. multilingual"
    )
    parser.add_argument(
        "--rope-factor",
        "--laya-rope-factor",
        dest="rope_factor",
        type=float,
        default=1,
        help="Experimental linear RoPE interpolation factor (default: 1, disabled)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
        help="KV cache element type; GGUF weights keep their own quantization",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp layer offload count; -1 for all, overrides --device",
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-request-branches", type=int, default=100)
    parser.add_argument(
        "--prefix-sharing",
        choices=["auto", "on", "off"],
        default="auto",
        help="Share prefix KV cells across branches; auto disables it for "
             "recurrent/hybrid architectures",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    host, port, model = args.pop("host"), args.pop("port"), args.pop("model")
    uvicorn.run(create_app(load_service(model, **args)), host=host, port=port)


if __name__ == "__main__":
    main()
