# Simple-JEV

Standalone classifier HTTP API using llama.cpp (GGUF) through
`llama-cpp-python`, with Vulkan GPU acceleration. Classifier validation,
prompt text, and response scoring come from the sibling `common/` folder. Keep
both folders in the checkout. No Open-JEV or vLLM runtime is required. The
package includes `common` when installed/built from this repo.

## Install and run

Use Python 3.12 or newer. `llama-cpp-python` compiles llama.cpp from source, so
a C/C++ toolchain (and the Vulkan SDK for GPU acceleration) must be present:

```bash
cd /path/to/simple-jev
python -m venv .venv
source .venv/bin/activate
# Build with the Vulkan backend for GPU acceleration:
CMAKE_ARGS="-DGGML_VULKAN=ON" pip install llama-cpp-python --no-cache-dir
# Or, for CPU-only execution, leave CMAKE_ARGS unset.
pip install -e './hf-server[test]'
simple-jev --model /path/to/model.gguf --device auto \
  --dtype bfloat16 --max-model-len 32768 \
  --max-batch-size 32 --max-batch-tokens 32768 --port 8000
```

On Windows, set the same variable with PowerShell: `$env:CMAKE_ARGS="-DGGML_VULKAN=ON"`.

Or run the file directly from the checkout:

```bash
python hf-server/hf_server.py --model /path/to/model.gguf --device cpu
```

After installation, `python -m hf_server` accepts the same arguments.
`--model` accepts a local `.gguf` file path, a directory containing exactly
one `.gguf` file, or a Hugging Face repository id. Use `--gguf-file NAME.gguf`
to select one weight file from a repository with multiple GGUF variants (the
repository identifier remains the request-visible model name).
Choose a context limit supported by the model. CPU testing can use
`--device cpu --dtype float32`.

`--device auto` offloads every layer to any llama.cpp backend compiled into the
wheel — Vulkan devices are used automatically when present, with CPU fallback.
`--device cpu` disables offload; `--n-gpu-layers N` sets an explicit layer count.
`--dtype` now selects the KV cache element type (`float16` default on GPUs,
`float32` for exact CPU scoring); GGUF weights keep their own quantization.

## API

See the [complete HTTP API reference](API_REFERENCE.md) for all request fields,
options, response formats, errors, limits, metrics and server arguments.

`POST /v1/classifier` and its alias `POST /v1/systemone` return non-streaming JSON.
`GET /health` reports readiness. `/docs` provides the generated API schema.
The request model must match the name/path passed to `--model`.

```json
{
  "model": "/path/to/model.gguf",
  "messages": [{"role": "user", "content": "Mia owns a red bicycle. Her dog is named Max."}],
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "dog": {"type": "noul", "instructions": "Is the dog named Max?"},
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Supported"]
    }
  }
}
```

Supply exactly one of `messages` or `state`. State accepts text or JSON.
Chat uses the model's GGUF chat template (`tokenizer.chat_template` metadata).
Choice and score support up to 50 entries. The server defaults to 100 scoring
branches per request; `--max-request-branches` configures the limit. The shared
v1 template uses exactly one branch per question. Legacy choice/score modes and
score-format switches are no longer accepted.
Invalid input returns readable 422 errors; a full queue returns 429.

Responses contain `model`, `answers`, and `usage`. Answers include confidence.
`usage.input_tokens` counts unique token prefixes once, not the entire shared
context once per question. `usage.output_tokens` is zero: this implementation
reads logits without sampling any output tokens. Set
`ENABLE_OPEN_JEV_ADVANCED_METRICS=1` to include detailed timing and metadata.
Standard completion settings such as `max_tokens` and `temperature` are ignored.

## Shared-prefix execution

The compiler calls `common.prepare_prompt(request, version="v1")`, assembles the
returned strings with state/chat roles, and applies the GGUF chat template.
See [the v1 specification](../common/PROMPT_STRUCTURE_V1.md). Compile each
question, find their exact common token prefix, and decode that prefix once on
sequence 0 of a unified llama.cpp KV pool. For each suffix batch, copy the
prefix cells into the batch's per-row sequences with `llama_memory_seq_cp`,
then pack all rows into one `llama_decode` call (no padding tokens) with logits
requested only at each row's final position. Row sequences are removed with
`llama_memory_seq_rm` after each batch, so cells and sequence ids stay reusable.
The prefix sequence remains untouched. Results return in request order.

Suffix batches are grouped longest-first, bounded by batch size, the packed
suffix token budget, and the remaining KV cells. Requests execute serially
against the context; parallelism is within each request. The pool is cleared
per request, with no persistent cross-request cache. The prefix prefill is one
decode and is not chunked by `--max-batch-tokens`.

## Precision notes

The CPU backend is the numerical reference: on it, shared-prefix scores equal
independent full-prompt decodes bit-for-bit (verified by the backend tests).
GPU backends add kernel-level floating-point differences: on the Vulkan builds
validated here, per-logit deviations up to roughly 0.1 versus CPU were observed
even with a `float32` KV cache, and `float16` KV adds its own rounding. Answers
rarely change, but exact parity with CPU or with the previous Transformers
implementation is **not** guaranteed on GPU. Use `--device cpu --dtype
float32` when scoring must be numerically anchored; treat GPU results as
approximately equal, not identical.

## Scope and validation

This reference currently accepts **text only**, including text messages.
Images, audio, video and tool calls are rejected. A multimodal GGUF does not
imply multimodal input support. Models must be GGUF files with a
`tokenizer.chat_template`, a non-recurrent/non-hybrid architecture (recurrent
state cannot be sequence-copied for shared prefixes), and single-token
rating/choice labels at the assistant boundary. Arbitrary model compatibility
is not guaranteed.

The shared v1 prompt and scoring rules are the source of truth for this server.
It does not claim exact numeric equivalence with another inference engine.
Tests verify the orchestration against a stubbed engine — prefix reuse, per-row
sequence copies, packed batches, cleanup, metrics, cancellation — and, when
`SIMPLE_JEV_GGUF` points at a local GGUF file, compare shared-prefix scores
against independent full-prompt decodes on the real engine plus an end-to-end
HTTP check. `SIMPLE_JEV_DEVICE` (default `cpu`) pins the device for those runs.
They establish execution equivalence, not answer quality.

```bash
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

## Laya backend

Install the optional SDK and select the backend explicitly:

```bash
pip install -e './hf-server[laya,test]'
USE_TF=0 python hf-server/hf_server.py \
  --backend laya --model convaiinnovations/laya --device cpu
# Add --subfolder multilingual or --subfolder typed-decisions for those checkpoints.
```

`--backend llama-cpp` remains the default; GGUF model commands are unchanged.
Laya loads once per process and uses its own
encoder/option-marker format, not the common v1 assistant-prefill template.
`--revision` selects the downloaded checkpoint revision. `--device auto` lets the
SDK select CUDA, MPS, or CPU; `--dtype` and the llama.cpp suffix batching
controls apply only to the GGUF backend. Laya uses its SDK precision policy and
batches the admitted questions together; `--max-request-branches` bounds that
batch.

The endpoint and `ClassifierRequest` stay the same. Send the repository ID as
`model`, including when a subfolder is selected at startup. Text `messages` are
serialized as a list of role/content objects. Images, tools, message extras, and
raw-logit diagnostics are rejected. Context overflow is rejected before inference;
the effective limit is the smaller of `--max-model-len` and the checkpoint's native
`max_len`. Laya's native formatter still budgets/truncates question and option text
according to its own `head_max_len` rules.

Choice/score probabilities retain the SDK's temperature scaling, confidence is
the largest returned probability, and Noul is its native binary positive-class
probability (not the v1 nine-bin mapping). SDK action fields are omitted. Token
usage sums the actual per-question sequences, so repeated context is counted;
output tokens are zero. Advanced metadata identifies the format as `laya-native`.
The shared admission queue and a model lock bound concurrent work; cancellation
cannot interrupt an already running engine forward.

Validation on macOS (2026-09-20): 50 common/server tests passed, including tiny
Qwen/Gemma backend regression tests and Laya adapter validation. Real weights for
`convaiinnovations/laya` and `Qwen/Qwen3.5-0.8B` both returned HTTP 200 through the
ASGI classifier route on CPU for a combined choice/score/Noul request, and rejected
an incorrect model ID with HTTP 422. Laya used 93 input tokens; Qwen used 751.
This validates integration, not accuracy: the small Qwen answered the example's
Noul question incorrectly. Direct MPS loading of Qwen crashed natively on this
Mac, so this run does not establish MPS compatibility. Laya's multilingual subfolder loading is covered by adapter tests, not real-weight
inference in this validation run. Typed Decisions extension validation is below.


### Experimental 2× Laya RoPE interpolation

For the ModernBERT Typed Decisions checkpoint:

```bash
USE_TF=0 python hf-server/hf_server.py \
  --backend laya --model convaiinnovations/laya \
  --subfolder typed-decisions --device cpu \
  --rope-factor 2 --max-model-len 2048
```

This halves full-attention and sliding-attention rotary inverse frequencies,
so position p has the original rotary angle at p/2. It doubles the checkpoint's
native sequence budget (1,024 → 2,048 here); the admission limit still respects
`--max-model-len`. It does not enlarge the sliding attention window. The flag
only supports unscaled ModernBERT RoPE and rejects other backends/architectures.
The default factor is 1, preserving existing model behavior. This is experimental:
long-input execution is not evidence of accuracy or calibration, and interpolation
also changes behavior on shorter inputs. No weights or downloaded configs are
modified; changes apply to the loaded process only.

Extension validation: all 52 tests passed. Real Typed Decisions weights with 2×
interpolation returned HTTP 200 for a 1,417-token sequence and chose the requested
refund category. An oversized sequence returned HTTP 422 at the 2,048-token limit.
This is an execution smoke test, not a long-context accuracy benchmark.


### General RoPE extension

`--rope-factor 2` enables experimental linear position interpolation for either
backend. The default is 1 (no change). Finite factors greater than 1, including
fractional factors (for example `--rope-factor 1.5`), are accepted. Rotary scaling
uses the exact factor; resulting token capacities are rounded down to integers.
`--laya-rope-factor` remains a CLI alias.

```bash
python hf-server/hf_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --gguf-file qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --device cpu --dtype float32 \
  --rope-factor 2 --max-model-len 2048
```

For the GGUF backend, the server configures llama.cpp's native linear rope
scaling on the context (`rope_scaling_type=LINEAR`, `rope_freq_scale=1/factor`),
which is the same scaling the `llama-server --rope-scale` flag applies. Laya
retains the ModernBERT implementation described above and doubles its checkpoint
sequence budget at 2×. `--max-model-len` remains the independent input admission
limit: RoPE scaling does not multiply this setting. These options are
experimental, not a promise that
all Hugging Face architectures support extended context or retain model quality.
