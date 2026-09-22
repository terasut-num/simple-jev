# CHANGED — GGUF / llama.cpp Refactor of the Classifier Server

Date: 2026-09-22
Scope: `hf-server/` inference engine replacement — Hugging Face Transformers + PyTorch → llama.cpp (GGUF) via `llama-cpp-python`, with Vulkan GPU acceleration.
Reference upstream: <https://github.com/featherless-ai/simple-jev/tree/main>

---

## 1. What was replaced

| Aspect | Before | After |
| --- | --- | --- |
| Inference engine | Transformers `AutoModelForCausalLM` / `AutoModelForImageTextToText` + PyTorch | llama.cpp via `llama-cpp-python` (`_internals.LlamaModel` / `LlamaContext`) |
| Model format | HF safetensors directories / repo ids | GGUF files, GGUF-containing directories, or HF GGUF repositories |
| Acceleration | `device_map` (CUDA/ROCm/CPU) | `n_gpu_layers` offload; Vulkan devices used automatically, CPU fallback |
| Shared-prefix reuse | `past_key_values` deep-copies + `reorder_cache` | Unified KV pool + `llama_memory_seq_cp` per row sequence |
| Suffix batching | Padded `[rows, width]` tensors + attention masks | Padding-free packed `llama_batch` rows |
| Tokenizer | `AutoTokenizer` | llama.cpp native GGUF vocabulary |
| Chat template | `tokenizer.apply_chat_template` | GGUF `tokenizer.chat_template` metadata rendered with llama-cpp-python's `Jinja2ChatFormatter` |
| KV precision | `torch.dtype` for weights and cache | `--dtype` selects KV cache element type only (weights keep GGUF quantization) |
| RoPE extension | Transformers `rope_parameters` config rewrite | llama.cpp context `rope_scaling_type=LINEAR`, `rope_freq_scale=1/factor` |

**Preserved without change** (core classification logic and API contracts):
- `common/` request validation, v1 prompt building, label semantics, softmax/scoring math, response envelope
- HTTP surface: `POST /v1/classifier`, alias `POST /v1/systemone`, `GET /health`, `/docs`; 422/429/499 error shapes and headers
- Metrics key names (`prefix_tokens`, `suffix_batch_sizes`, `engine_forwards`, `branch_prompt_tokens`, `computed_prompt_tokens`, `logical_prefill_tokens`, `padded_suffix_tokens`, `branch_output_tokens=0`, `scored_positions`, `backend_seconds`, `queue_seconds`, `total_seconds`, `prefill_strategy="shared_prefix"`)
- Usage accounting: unique token-prefix union for `input_tokens`, `output_tokens` always 0
- Advanced-metrics opt-in (`ENABLE_OPEN_JEV_ADVANCED_METRICS`, `raw_logits`)
- Queue/concurrency model: concurrency 1, 16 queued slots, cooperative cancellation between batches under the model lock
- `PromptCompiler`'s duck-typed tokenizer interface (`encode(text, add_special_tokens=False)`, `apply_chat_template(..., tokenize=False)`) — external consumers such as `RFDT/train.py` and `common/tests/test_pipeline.py` are unaffected
- The optional Laya backend (`--backend laya`) including its RoPE extension

---

## 2. New backend design (`LlamaCppBackend`)

The shared-prefix execution strategy was translated to llama.cpp's multi-sequence API:

1. **Per request** the KV pool is cleared (`llama_memory_clear`), so no state leaks between callers.
2. The branches' exact common token prefix is decoded **once, unchunked** on sequence 0 (`llama_batch` with no logits requested).
3. For each suffix batch: `llama_memory_seq_cp(memory, 0, row+1, 0, len(prefix))` marks the shared prefix cells for each row's own sequence id — attention then sees prefix + suffix without recomputing prefill.
4. All rows of the batch are packed into **one** `llama_decode` call: no padding tokens, explicit `pos` continuing after the prefix, `logits[i]` flagged only at each row's final suffix token.
5. `llama_get_logits_ith` is read at each row's flagged batch position; only the permitted label token ids are converted to plain Python floats, keyed by the plan's exact output labels.
6. `llama_memory_seq_rm(memory, row+1, 0, -1)` after each batch drops the rows' claims on shared cells and frees their suffix cells, making both cells and sequence ids reusable for the next batch.

Batching keeps the previous engine's shape: suffixes sorted longest-first, rows bounded by `--max-batch-size`, by `max_batch_tokens // width` (width of the longest suffix in the batch), and by the remaining KV cell budget `(n_ctx - prefix) // width`.

Backend results changed from 1-D torch tensors to **label→float mappings** per branch, one of the forms `common.response_scoring` already accepts, removing the torch dependency from the scoring path.

Context parameters chosen by the loader:

| Parameter | Value | Why |
| --- | --- | --- |
| `n_ctx` | `--max-model-len` (clamped to `n_ctx_train`) | admission limit and KV pool size |
| `n_batch` | `max(max_batch_tokens, max_model_len)` | one decode must hold the prefix or a full token budget; llama.cpp additionally clamps it to `n_ctx` |
| `n_seq_max` | `max_batch_size + 1` | sequence 0 (prefix) plus one row sequence per batch slot |
| `kv_unified` | `True` | partial-range `seq_cp` requires the unified cache; the split cache asserts |
| `type_k` / `type_v` | `--dtype` mapped | KV cache element type |
| `rope_*` | from `--rope-factor` | linear position interpolation |

---

## 3. Empirical findings (probe work before implementation)

The engine swap was preceded by probe scripts against a real GGUF to pin down llama.cpp semantics. These findings drove the design:

1. **CPU is the numerical reference.** On CPU, shared-prefix scoring (`seq_cp` + packed suffix decode) equals independent full-prompt decodes **bit-for-bit** (max diff 0.0). Chunked decode equals single-shot decode exactly on CPU as well.
2. **Vulkan kernel divergence.** This build's Vulkan backend deviates from CPU by up to ~0.1 logits per token for multi-chunk decodes (prefill-then-continue), regardless of flash-attention on/off and KV dtype (f16 or f32). Single-decode-per-batch scoring is therefore used throughout; remaining GPU-vs-CPU differences are documented as a precision caveat. Live CPU and Vulkan runs produced agreeing answers (see §6).
3. **`llama_get_logits_ith(ctx, i)` indexes batch token positions**, not output ordinals. `i` must be a position flagged in the batch (`batch.logits[i] == true`), otherwise the call returns NULL and ctypes raises.
4. **Partial-range `llama_memory_seq_cp` requires `kv_unified=True`.** The default split cache aborts with `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`.
5. **Unified KV pool = `n_ctx` cells total**, shared across sequences by per-cell sequence bitmasks. `n_batch` is silently clamped to `n_ctx`. Requested `n_ctx` is rounded up by llama.cpp (e.g. 64 → 256; 1000 → 1024); the backend reads `context.n_ctx()` for the real budget.
6. **Per-sequence positions must remain consecutive across decodes** (`Y = X + 1`). Re-decoding a sequence from position 0 without clearing fails; independent reference decodes need fresh sequence ids or a clear.
7. **`llama_batch` memory is recycled and uninitialized.** Every field of every used slot must be set; failures surface as misleading "invalid token[i] = <garbage>" messages. Batch arrays are filled through cached ctypes handles to avoid repeated pointer fetches.
8. **`n_outputs_max_per_seq` defaults to 1** — one flagged output per sequence per decode is the safe contract for multi-row scoring batches.
9. **ggml_type values**: F32=0, F16=1, BF16=30. The constants are not re-exported by `llama_cpp.py`, so `KV_CACHE_TYPES = {"float32": 0, "float16": 1, "bfloat16": 30}` maps `--dtype` to `type_k`/`type_v`.
10. **Recurrent/hybrid architectures cannot share prefixes** (`llama_memory_seq_cp` cannot copy their state). The loader rejects them via `llama_model_is_recurrent` / `llama_model_is_hybrid`.
11. **GGUF chat templates** live in `tokenizer.chat_template` metadata and render correctly with llama-cpp-python's `Jinja2ChatFormatter` (trim_blocks/lstrip_blocks and the `{% generation %}` pass-through match HF). The formatter bakes `add_generation_prompt` in at construction, so one cached formatter is built per flag value; `enable_thinking` and other HF-style kwargs are forwarded into the render, mirroring Transformers.
12. **Tokenization parity**: `LlamaModel.tokenize(text_bytes, add_bos, special)` with `add_bos=False` mirrors HF `encode(add_special_tokens=False)` when the rendered template already carries its special tokens.

---

## 4. Files changed

### `hf-server/hf_server.py` — the core refactor

- **Removed**: `HFBackend`, all torch/Transformers imports, `configure_rope`'s Transformers implementation, the Transformers branch of `load_service`, `copy`/`inspect` imports.
- **Added**:
  - `LlamaCppBackend` — engine described in §2, with the same async/cancellation/locking contract as before (`asyncio.to_thread` + cooperative stop event + thread lock held through worker exit).
  - `LlamaCppTokenizer` — GGUF-native `encode` / `apply_chat_template` adapter for `PromptCompiler`.
  - `resolve_gguf_path` — local `.gguf` file, directory with exactly one `.gguf` (shard sets count as one), or HF repository download via `huggingface_hub`; `--gguf-file` selects one file from a multi-variant repository and avoids downloading every quantization. The repository id remains the request-visible model name.
  - `resolve_n_gpu_layers` — maps `--device` (`cpu` → 0 layers; `auto`/`gpu`/`vulkan`/… → all layers) with `--n-gpu-layers` overriding.
  - `KV_CACHE_TYPES` — dtype → ggml_type mapping for the KV cache.
  - `configure_rope` (rewritten) — llama.cpp linear rope scaling on the context params.
- **`load_service`**: `backend` default is now `"llama-cpp"`; the GGUF branch loads model + context with the parameters from §2, rejects recurrent/hybrid models and unsupported dtypes/devices, and reports metadata: `backend`, `model_revision`, `gguf_path`, `n_gpu_layers`, `kv_cache_dtype`, `rope_factor`. The Laya branch is unchanged.
- **`main()`**: `--backend` choices are `["llama-cpp", "laya"]`; new `--n-gpu-layers`; `--dtype` help documents its KV-cache meaning. `--device`/`--dtype`/`--max-batch-tokens` keep their names for command compatibility.
- Metrics: `backend` is `"llama-cpp"`; because rows are packed without padding, `computed_prompt_tokens == logical_prefill_tokens` and `padded_suffix_tokens` equals the packed suffix total (key names retained for API stability, meanings documented).

### `hf-server/pyproject.toml`

- Dependencies: `torch`, `transformers`, `accelerate` → `llama-cpp-python>=0.3.35`; `jinja2` added explicitly (used by the chat formatter). Description updated. Laya and test extras unchanged.

### Tests

- **Removed** `hf-server/tests/test_hf_backend.py` (torch/tiny-Transformers-model regression suite).
- **Added** `hf-server/tests/test_llama_backend.py`:
  - Stubbed-engine orchestration tests (no weights): prefix decoded once on seq 0, exact per-row `seq_cp` ranges, packed rows with one flagged position each, `seq_rm` cleanup and sequence-id reuse, width-based batch splitting, identical-prompt suffix reservation, decode-failure and plan-required errors, cancellation contract, metrics values.
  - Opt-in real-engine tests gated by `SIMPLE_JEV_GGUF` (local GGUF path) and `SIMPLE_JEV_DEVICE` (default `cpu`): backend shared-prefix scores vs independent full-prompt decodes on the same context, plus an end-to-end HTTP check through `create_app` covering choice/score/noul, usage, and answer shape.
- **Updated** `hf-server/tests/test_laya.py`: removed the Transformers-specific `configure_rope` tests; added `test_llama_cpp_loader_rejects_bad_inputs`, `test_llama_cpp_rope_configuration`, `test_resolve_gguf_paths`; added a skip guard for the torch-dependent Laya RoPE numerical test (torch remains optional for the Laya path only).

### Documentation

- `hf-server/README.md`: GGUF/Vulkan install instructions (`CMAKE_ARGS="-DGGML_VULKAN=ON"`), model resolution rules, rewritten shared-prefix execution section, new **Precision notes** section, scope and validation updates, Laya/RoPE sections updated.
- `hf-server/API_REFERENCE.md`: backend/metadata/metrics tables (`llama-cpp`, new metadata fields, packing-aware metric meanings), batching/limits section, GPU precision note, full CLI argument table including `--n-gpu-layers` and the new `--device`/`--dtype` semantics.
- Root `README.md`: intro, "Running the GGUF Server" instructions, GPU example (Vulkan offload instead of `--device cuda`), explicit `--gguf-file` selection for multi-variant Hugging Face repositories, example model identifiers, component table, prompt-flow description, and testing/model-requirements sections.

---

## 5. CLI changes

| Flag | Change |
| --- | --- |
| `--backend` | choices `llama-cpp` (new default), `laya`; `transformers` removed |
| `--model` | now a GGUF path / GGUF directory / HF GGUF repository id |
| `--device` | `cpu` = no offload; `auto`/`gpu`/`vulkan` = offload all layers via any compiled llama.cpp backend (Vulkan first when present); unknown values rejected |
| `--dtype` | selects the KV cache element type (float32/float16/bfloat16); weights keep their GGUF quantization |
| `--n-gpu-layers` | new; explicit llama.cpp offload count, overrides `--device` |
| `--revision` | used for the HF GGUF download |
| `--gguf-file` | new; selects one `.gguf` from a Hugging Face repository with multiple variants, downloading only that file |
| others (`--max-model-len`, `--max-batch-size`, `--max-batch-tokens`, `--max-request-branches`, `--rope-factor`, `--subfolder`, `--host`, `--port`) | names unchanged; `--max-model-len` additionally sizes the KV pool and is clamped to the trained context |

---

## 6. Validation results

- **Test suite**: `python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q` → **62 passed, 2 skipped** by default (the 2 real-engine tests skip without `SIMPLE_JEV_GGUF`).
- **Real-engine tests** (with `SIMPLE_JEV_GGUF` pointing at a local Qwen2.5-0.5B-Instruct Q2_K GGUF): **64 passed** on CPU, including the shared-prefix-vs-independent fidelity test (CPU worst diff < 1e-3, measured 0.0) and the end-to-end HTTP test; the HTTP test also passed with `SIMPLE_JEV_DEVICE=vulkan`.
- **Live server, CPU** (`--device cpu --dtype float32 --max-model-len 2048`): combined choice/score/noul request returned HTTP 200 with `choice: "red"` (confidence 0.950), score 0.919 ("Fully supported"), `usage.input_tokens: 715`, `output_tokens: 0`; wrong model id → 422; `/health` → 200.
- **Live server, Vulkan** (`--device auto --dtype float16`): all 24 layers offloaded to the NVIDIA RTX GPU (`Vulkan1` KV buffer, flash attention enabled); the same request returned `choice: "red"` (0.948), score 0.9194 — agreeing answers with small GPU-vs-CPU logit differences consistent with the documented precision caveat.
- `--help` still loads no weights (heavy imports remain local to `load_service`).

### Test-environment notes

- The project `.venv` (Python 3.14) was extended with a prebuilt Vulkan-enabled `llama-cpp-python` 0.3.35 wheel (contains `ggml-vulkan.dll`) plus its `diskcache` dependency. Building from source requires CMake + MSVC and `CMAKE_ARGS="-DGGML_VULKAN=ON"`; no C toolchain is present on this machine, so the prebuilt wheel was reused.
- A temporary `.probe/` directory used for engine-mechanics validation was removed after the refactor; nothing in the package depends on it.

---

## 7. Known caveats

1. **GPU precision**: exact parity with CPU (and with the previous Transformers implementation) is not guaranteed on GPU backends; this build's Vulkan kernels showed up to ~0.1 per-logit deviation for chunked decodes even with an f32 KV cache. Use `--device cpu --dtype float32` when scoring must be numerically anchored. Answers agreed in all live validation runs.
2. **Model requirements**: GGUF with `tokenizer.chat_template`, non-recurrent/non-hybrid architecture, and single-token-stable answer labels at the assistant boundary (checked per branch by `PromptCompiler` as before). GGUFs lacking a chat template are rejected at load.
3. **Laya**: unchanged, still requires its optional SDK and (for the RoPE numerical test) torch; both install extras remain.
4. **`--backend transformers` is gone**: scripts passing it will now get an `Unknown backend` error.