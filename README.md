# Simple Jev Project with llama.cpp and GGUF implementation

Use compatible open models from huggingface, for structured classification and scoring, without training a separate classifier head.

Explore the demos, playground, and documentation at [simple-jev.featherless.ai](https://simple-jev.featherless.ai/).

Send shared context and a set of questions. Simple Jev reads the model's next-token logits for each question and builds a JSON response containing choices, rubric scores, or truth/support judgments. The model does not generate a JSON completion: the server constructs the response from the scores.

The current implementation runs locally with llama.cpp (GGUF models) through
`llama-cpp-python`, with Vulkan GPU acceleration. Shared request validation,
versioned prompt instructions, and response scoring live in the plain Python
`common/` folder so other inference implementations can use the same rules.

## Try it today

Try the [public demo API](https://simple-jev-demo-api.featherless.ai/v1/) with **no login, API key, or authentication required**. The demo has a **2k-token context limit** and is **rate limited to 2 requests per second (2 RPS)**.

First, list the available models with `GET /v1/models`:

```bash
curl https://simple-jev-demo-api.featherless.ai/v1/models
```

The demo already serves Gemma as `featherless-ai/gemma-4-26B-A4B-classifier`. Try it directly with `POST /v1/classifier`, or use another model ID returned by the list:

```bash
curl https://simple-jev-demo-api.featherless.ai/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "featherless-ai/gemma-4-26B-A4B-classifier",
  "state": "Mia owns a red bicycle.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    }
  }
}
JSON
```

For production deployments, [Featherless paid plans](https://featherless.ai/) offer higher limits. To run the server yourself, follow the setup below.

## Running the GGUF Server

Use Python 3.12 or newer. The commands below use Python 3.13.

```bash
# Clone the repository and create an environment.
git clone https://github.com/featherless-ai/simple-jev.git
cd simple-jev
python3.13 -m venv .venv
source .venv/bin/activate

# Build llama.cpp with the Vulkan backend for GPU acceleration + Administrator level
# (requires a C/C++ toolchain and the Vulkan SDK). Omit for CPU-only.
set CMAKE_ARGS="-DGGML_VULKAN=ON"
set CMAKE_GENERATOR=Visual Studio 17 2022
python -m pip install llama-cpp-python --no-cache-dir

# Install the server, including the shared common modules.
python -m pip install -e './hf-server'

# Start with a small GGUF model on CPU.
python hf-server/hf_server.py --model Qwen/Qwen2.5-0.5B-Instruct-GGUF --gguf-file qwen2.5-0.5b-instruct-fp16.gguf --device cpu --dtype float32 --max-model-len 4096 --max-batch-size 4 --max-batch-tokens 4096

# Alternatively, offload every layer to any available GPU backend
# (Vulkan devices are used automatically when the wheel supports them).
python hf-server/hf_server.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --gguf-file qwen2.5-0.5b-instruct-fp16.gguf \
  --device auto --dtype bfloat16 \
  --max-model-len 8192 \
  --max-batch-size 4 --max-batch-tokens 8192

# Alternatively, run Laya Typed Decisions with its native encoder backend.
# Stop the previous server first, or choose a different --port.
python -m pip install -e './hf-server[laya]'
USE_TF=0 python hf-server/hf_server.py \
  --backend laya \
  --model convaiinnovations/laya \
  --subfolder typed-decisions \
  --device cpu \
  --rope-factor 2 --max-model-len 2048
```

`--model` accepts a local `.gguf` file, a directory containing exactly one
`.gguf`, or a Hugging Face repository id. For repositories with multiple GGUF
variants, select one with `--gguf-file NAME.gguf`; only that file is downloaded
on first run, and the repository identifier remains the request-visible model
name. For CPU-only builds of llama.cpp, omit `CMAKE_ARGS`.

The GPU example offloads all layers through llama.cpp's device enumeration:
with a Vulkan-enabled wheel, weights, KV cache, and flash-attention kernels run
on the GPU, with CPU fallback when no device is present. Allow memory for the
full model weights, KV cache, and inference buffers; sparse expert activation
does not mean only the active experts occupy memory. This is a launch example,
not a verified full-size benchmark. GPU kernels introduce small floating-point
differences relative to CPU; use `--device cpu --dtype float32` when scoring
must be numerically anchored.

CPU execution verifies deterministic llama.cpp scoring, but does not guarantee
bit-for-bit agreement with the original PyTorch server. `--dtype float32`
selects the llama.cpp KV-cache precision only; it does not convert GGUF model
weights. For example, `qwen2.5-0.5b-instruct-fp16.gguf` retains FP16 weights,
while the original Hugging Face checkpoint may load with a different weight
precision. Different kernels, weight formats, and accumulation behavior can
therefore shift raw logits and their softmax probabilities slightly even when
the prompt, token accounting, and selected answer are the same.

The Laya example loads the specialized Typed Decisions checkpoint. Use `--device cuda` for an NVIDIA GPU, and send `"model": "convaiinnovations/laya"` in API requests. Its default native limit is 1,024 tokens per question. This example explicitly enables experimental 2× linear RoPE interpolation and a 2,048-token sequence budget, including instructions, options, and state. Both full and sliding attention rotary frequencies are halved; the local attention window is unchanged. This enables longer inputs but does not establish accuracy or calibration beyond the checkpoint's training length. Omit `--rope-factor 2` to retain the native behavior. See [Laya backend details](hf-server/README.md#laya-backend).

The server listens on `http://127.0.0.1:8000`. Once the model is loaded:

```bash
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. After installation, `simple-jev` and `python -m hf_server` accept the same arguments as the script.

## How do I use the API?

Send a non-streaming `POST /v1/classifier` request. The `model` value must exactly match the ID or path used to start the server. The examples below use Qwen; if you started Gemma, use `google/gemma-4-26B-A4B-it` instead. Supply exactly one of:

- `state`: a string, JSON object, or JSON array containing the shared context.
- `messages`: text chat history, rendered using the model's own chat template.

The API takes inspiration from TypeSafe's structured-decision interface and includes project-specific behavior. `/v1/systemone` is an alias of `/v1/classifier`; both run the same implementation. Use this repository's [API reference](hf-server/API_REFERENCE.md) as the contract for clients.

```bash
curl http://127.0.0.1:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "state": "Mia owns a red bicycle. Her dog is named Max.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Partially supported", "Fully supported"]
    },
    "dog": {
      "type": "noul",
      "instructions": "Is Mia's dog named Max?"
    }
  }
}
JSON
```

Question IDs become keys in `answers`.

### CPU comparison with the original PyTorch server

The following results use the same request. The original server ran the
Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` checkpoint with PyTorch:

```json
{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "answers": {
        "color": {
            "type": "choice",
            "confidence": 0.877018392086029,
            "probabilities": {
                "red": 0.877018392086029,
                "blue": 0.12298166006803513
            },
            "choice": "red"
        },
        "support": {
            "type": "score",
            "confidence": 0.558726966381073,
            "probabilities": {
                "0": 0.05434797331690788,
                "1": 0.3869251012802124,
                "2": 0.558726966381073
            },
            "score": 1.5043790340423584,
            "legend": {
                "0": "Unsupported",
                "1": "Partially supported",
                "2": "Fully supported"
            }
        },
        "dog": {
            "type": "noul",
            "noul": 0.010005198717117306
        }
    },
    "usage": {
        "input_tokens": 743,
        "output_tokens": 0
    }
}
```

The llama.cpp server ran
`qwen2.5-0.5b-instruct-fp16.gguf` on CPU with `--dtype float32`:

```json
{
  "model": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
  "answers": {
    "color": {
      "type": "choice",
      "confidence": 0.8988586068153381,
      "probabilities": {
        "red": 0.8988586068153381,
        "blue": 0.10114137083292007
      },
      "choice": "red"
    },
    "support": {
      "type": "score",
      "confidence": 0.6025716662406921,
      "probabilities": {
        "0": 0.03482682630419731,
        "1": 0.36260151863098145,
        "2": 0.6025716662406921
      },
      "score": 1.5677448511123657,
      "legend": {
        "0": "Unsupported",
        "1": "Partially supported",
        "2": "Fully supported"
      }
    },
    "dog": {
      "type": "noul",
      "noul": 0.010013843774795523
    }
  },
  "usage": {
    "input_tokens": 743,
    "output_tokens": 0
  }
}
```

Both servers selected `red`, placed `support` toward the third rubric level,
and reported 743 input tokens. Their probabilities and score differ slightly,
as expected from the FP16 GGUF weights and differing PyTorch/llama.cpp numeric
implementations described above.

| Question type | Input criteria | Result |
| --- | --- | --- |
| `choice` | Object with 2–50 candidate IDs and optional descriptions | Highest-probability candidate, its confidence, and the candidate distribution. |
| `score` | Array of 2–50 rubric levels, lowest to highest | Expected zero-based rubric index, confidence, distribution, and rubric legend. A three-level rubric returns a value from 0 to 2, including fractional values. |
| `noul` | Optional `true` and/or `false` descriptions | Truth/support judgment from 0.01 to 0.99, derived from the model's distribution over nine rating tokens. |

Choice and score confidence is the largest probability among their allowed labels. These distributions, and the Noul value, are not calibrated probabilities of correctness.

For chat input, send `messages` instead of `state`. This complete example classifies a customer message and provides explicit descriptions for the possible answers:

```bash
curl http://127.0.0.1:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3.5-0.8B",
  "messages": [
    {"role": "user", "content": "I was charged twice for my subscription. Please refund the duplicate charge."}
  ],
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "Payments, invoices, and refunds",
        "technical": "Errors and problems using the product"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly request a refund?",
      "criteria": {
        "true": "The customer asks for money back.",
        "false": "The customer makes no refund request."
      }
    }
  }
}
JSON
```

Unknown top-level request fields are ignored, including completion settings such as `temperature`, `max_tokens`, and `stream`. Unknown fields inside questions and options are rejected. There is no completion sampling or streaming. The GGUF server currently supports text only; images, audio, video, and tool calls are unsupported.

`usage.input_tokens` counts unique token prefixes within the request, sharing the common context across questions. `usage.output_tokens` is zero because no output tokens are generated. For diagnostic timings, start the server with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1`; adding `"options": {"raw_logits": true}` to a request then includes selected-token logits.

## What is a classifier, and why “System One”?

A classifier maps input to a defined set of answers. For example, a support system might route a message to `billing` or `technical`, score its urgency against an ordered rubric, and judge whether it requests a refund. Those decisions can feed directly into ordinary application code.

TypeSafe uses “System One” to describe models designed for fast, structured decisions, drawing the name from the distinction between fast intuitive thinking and slower deliberate reasoning. Its [introduction to System One and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) explains that motivation. Simple Jev explores this style of interface using existing open language models. It does not reproduce TypeSafe's model architecture or training, or establish equivalent accuracy, calibration, or speed.

In this implementation, the useful change is how the model is used:

1. The shared prompt builder creates consistent classifier instructions and one scoring branch per question.
2. The server renders those instructions and the context into the model's native chat format from the GGUF's chat template.
3. It decodes the exact common token prefix once and copies those KV cells into per-question sequences for each suffix batch.
4. It reads the next-token logits for the allowed answer labels. Shared scoring code normalizes those scores and constructs the JSON response.

This avoids generating and parsing a prose or JSON answer token by token. Reusing the context can also reduce repeated computation when several questions refer to the same input. Actual latency depends on the model, hardware, context length, and number of questions. Cache reuse currently lasts only for a single request.

A valid response structure does not guarantee a correct decision. Model capability and question wording still matter; evaluate answer quality on your own task separately from validating the server's inference and response pipeline.

## How shared prompts and prefill-only scoring reduce work

A normal text-generation request has a **prefill** step that processes the input, followed by **decode** steps that generate tokens one at a time. Prefill already produces logits for the next token. Simple Jev uses those logits directly to score predefined answer labels, so it needs no autoregressive decode loop.

For several questions about the same context, most of the prompt is identical. After applying the model's chat template and tokenizing each question's prompt, the server finds their exact common token prefix:

```text
Shared instructions + context + shared question briefing
                           │
                     Prefill once
                     Save KV cache
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  Color question    Support question    Dog question
         │                 │                 │
    Label logits      Label logits      Label logits
         └─────────────────┼─────────────────┘
                           ▼
                JSON built by the server
```

The KV pool stores the model's attention state for the shared prefix. Each question continues from a copy of those cells with its own suffix and answer prefix. The server packs these suffixes into batched decodes, reads the logits at each row's last real token, and passes the selected label scores to the shared response scorer. Questions do not consume one another's answers.

For example, suppose four question prompts each contain a 1,000-token common prefix and a 50-token suffix:

| Execution | Prompt tokens processed, excluding padding |
| --- | --- |
| Evaluate each complete prompt separately | `4 × (1,000 + 50) = 4,200` |
| Reuse the shared prefix | `1,000 + 4 × 50 = 1,200` |

If all four suffixes fit in one batch, the shared execution takes one prefix decode and one batched suffix decode. These token counts illustrate avoided repeated input processing, not a measured latency ratio: each suffix still attends to the cached prefix, and KV copies and model execution have costs.

`--max-batch-size` limits questions per suffix batch; `--max-batch-tokens` limits the number of packed suffix tokens in that batch. Neither limits total model/cache memory or chunks the shared prefix. The current server reuses the KV pool within a request and processes model requests serially. See the [GGUF execution guide](hf-server/README.md#shared-prefix-execution) for details.

## Shared prompt contract and project layout

| Location | Purpose |
| --- | --- |
| [`common/`](common/README.md) | Plain Python modules for `ClassifierRequest`, prompt planning, and response scoring. No separate package installation is required. |
| [`common/PROMPT_STRUCTURE_V1.md`](common/PROMPT_STRUCTURE_V1.md) | Language-independent v1 specification: inputs, prompt strings, chat roles, answer labels, and scoring rules. |
| [`hf-server/hf_server.py`](hf-server/hf_server.py) | Single-file llama.cpp implementation: GGUF loading, chat rendering, shared-prefix inference, HTTP API, and CLI. |
| [`hf-server/API_REFERENCE.md`](hf-server/API_REFERENCE.md) | Detailed request/response contract, validation, diagnostics, and configuration. |
| [`RFDT/`](RFDT/README.md) | Task-specific decision training: prepare labels, distill teacher estimates, train on answer-token logits, and export a student. |

`prepare_prompt(request, version="v1")` returns a cacheable system prompt prefix, prefix instruction, suffix instruction, and ordered questions. The inference implementation handles chat formatting and model execution; `common/response_scoring.py` converts label logits or mapped PyTorch tensors into answers.

The version fixes one prompt/scoring configuration so implementations can stay consistent, including implementations in other languages. It defaults to `v1`; the HTTP API currently uses that version. There are no per-request independent/rating modes or score-format switches.

## Testing

From the repository root, with the environment activated:

```bash
python -m pip install -e './hf-server[test]'
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

The tests cover request validation, prompt construction, response scoring, tensor/token mapping, HTTP behavior, and shared-prefix-versus-full-prompt inference. The GGUF backend's orchestration tests run against a stubbed engine without any weights; its real-engine tests are opt-in via `SIMPLE_JEV_GGUF` pointing at a local GGUF file. They do not measure classification accuracy.

Models need a GGUF file with a usable chat template, a non-recurrent architecture whose KV state can be shared across sequences, and answer labels that each extend the rendered prompt by exactly one distinct token. The server checks label tokenization; compatibility with every open model is not guaranteed.

## One more thing: Really Fancy Decision Training (RFDT)

Want a smaller model that is better at your specific use case? **RFDT** lets you fine-tune a model on the decisions your application needs. Provide context or chat history, questions, and answers—or let a larger teacher model supply the missing answers.

RFDT trains directly on the allowed answer-token logits using the same prompt structure as Simple Jev inference. The scripts support dataset preparation, teacher labeling, multi-GPU training, LoRA adapters, evaluation, and export to the HF server. See the [RFDT guide and examples](RFDT/README.md) to get started on your own hardware.

As we scale up support and usage of Simple Jev models on [Featherless](https://featherless.ai/), we will roll out support for serving fine-tuned models and running fine-tuning on the platform. The RFDT scripts are available in this repository today; hosted fine-tuned model support and fine-tuning are part of that upcoming rollout.
