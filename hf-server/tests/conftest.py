"""Shared lightweight test doubles for HF service and prompt tests.

FakeCompiler uses the real common plan but a synthetic two-token shared prefix,
so usage expectations are exact and independent of any downloaded tokenizer.
Tokenizer maps UTF-8 bytes to IDs and exposes an inspectable role-based template.
It exercises adapter control flow, not production tokenizer/model compatibility.
"""

from hf_server import Branch, CompiledRequest

from common import prepare_prompt


class FakeCompiler:
    """Small deterministic token tree for service/usage tests without inference."""

    def compile(self, request):
        """Give each real plan question a synthetic [1, 2, unique_leaf] path."""
        plan = prepare_prompt(request)
        return CompiledRequest(
            plan,
            [
                Branch(
                    q.branch_id,
                    [1, 2, i + 3],
                    list(range(len(q.output_labels))),
                    [],
                    q.answer_prefix,
                )
                for i, q in enumerate(plan.questions)
            ],
        )


class Tokenizer:
    """Make labels single-byte tokens and expose exactly which roles are rendered."""

    def encode(self, text, **kwargs):
        """Return deterministic byte IDs; keyword arguments mimic the HF API."""
        return list(text.encode("utf8"))

    def apply_chat_template(self, messages, **kwargs):
        """Require the intended generation settings and open an assistant turn."""
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            + "\nassistant: "
        )


# GGUF header values llama.cpp's converter writes for the Qwen3.5 4B backbone
# (Qwen/Qwen3.5-4B text config); the vocabulary size comes from the loaded model.
QWEN35_4B_GGUF = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 32,
    "qwen35.embedding_length": 2560,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.feed_forward_length": 9216,
}


def fake_gguf_loader(monkeypatch, tokenizer, metadata=None, n_vocab=248320):
    """Replace llama.cpp loading with recorders; return the created objects.

    load_service still runs its real control flow (path resolution, header
    fingerprinting, vocabulary-only probe, weight load, context parameters),
    but models/contexts are plain records and every LlamaCppTokenizer is the
    supplied test tokenizer. Records carry the llama.cpp parameters used.
    """
    import types

    import llama_cpp
    from llama_cpp import _internals

    import hf_server

    created = types.SimpleNamespace(models=[], contexts=[], tokenizers=[], paths=[])

    class Model:
        def __init__(self, *, path_model, params, verbose):
            self.path = path_model
            self.vocab_only = bool(params.vocab_only)
            self.n_gpu_layers = params.n_gpu_layers
            self.model = self
            self.closed = False
            created.models.append(self)

        def n_vocab(self):
            return n_vocab

        def n_ctx_train(self):
            return 262144

        def close(self):
            self.closed = True

    class Context:
        def __init__(self, *, model, params, verbose):
            self.model = model
            self.params = params
            created.contexts.append(self)

    def resolve(model_name, revision, gguf_file=None):
        created.paths.append((model_name, revision, gguf_file))
        return model_name

    def make_tokenizer(model, metadata=None, chat_template=None):
        created.tokenizers.append((model, chat_template))
        return tokenizer

    monkeypatch.setattr(hf_server, "resolve_gguf_path", resolve)
    monkeypatch.setattr(hf_server, "read_gguf_metadata", lambda path: dict(metadata or {}))
    monkeypatch.setattr(hf_server, "LlamaCppTokenizer", make_tokenizer)
    monkeypatch.setattr(_internals, "LlamaModel", Model)
    monkeypatch.setattr(_internals, "LlamaContext", Context)
    monkeypatch.setattr(llama_cpp, "llama_model_is_recurrent", lambda model: False)
    monkeypatch.setattr(llama_cpp, "llama_model_is_hybrid", lambda model: False)
    return created
