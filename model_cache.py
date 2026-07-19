"""
Process-level model cache.

HuggingFace embedding models take 15-40s to load from disk on first use.
Caching the instance here means it's loaded ONCE per Flask process and
reused across all subsequent requests — making repeat queries nearly instant.
"""

import threading

_embed_models: dict = {}
_mistral_embedding_lock = threading.RLock()

# Decoder-only models like microsoft/harrier-oss-v1-270m require a task
# instruction prefix on queries (but NOT on documents) for best retrieval.
_QUERY_INSTRUCTIONS = {
    # [ORIGINAL] harrier query instruction — commented out while testing BGE-M3
    # "microsoft/harrier-oss-v1-270m": (
    #     "Given a search query, retrieve relevant passages that answer the query: "
    # ),

    # [TEST] BAAI/bge-m3 uses a short retrieval prefix on queries.
    "BAAI/bge-m3": (
        "Represent this sentence for searching relevant passages: "
    ),
}


def get_embed_model(model_name: str):
    """Return the cached embedding client for the configured provider."""
    if model_name not in _embed_models:
        from config import Config

        if model_name == Config.MISTRAL_EMBED_MODEL:
            from llama_index.embeddings.mistralai import MistralAIEmbedding

            class SerializedMistralAIEmbedding(MistralAIEmbedding):
                """Prevent overlapping embedding calls across Flask threads."""

                def _get_query_embedding(self, query, *args, **kwargs):
                    with _mistral_embedding_lock:
                        return super()._get_query_embedding(query, *args, **kwargs)

                def _get_text_embedding(self, text, *args, **kwargs):
                    with _mistral_embedding_lock:
                        return super()._get_text_embedding(text, *args, **kwargs)

                def _get_text_embeddings(self, texts, *args, **kwargs):
                    with _mistral_embedding_lock:
                        return super()._get_text_embeddings(texts, *args, **kwargs)

            print(f"[model_cache] Creating Mistral embedding client: {model_name} ...")
            _embed_models[model_name] = SerializedMistralAIEmbedding(
                model_name=model_name,
                api_key=Config.MISTRAL_API_KEY,
                embed_batch_size=Config.EMBED_BATCH_SIZE,
            )
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            print(f"[model_cache] Loading HuggingFace embed model: {model_name} ...")
            kwargs = dict(
                model_name=model_name,
                trust_remote_code=True,
                embed_batch_size=Config.EMBED_BATCH_SIZE,
            )
            query_instr = _QUERY_INSTRUCTIONS.get(model_name)
            if query_instr:
                kwargs["query_instruction"] = query_instr
                kwargs["text_instruction"] = ""
            _embed_models[model_name] = HuggingFaceEmbedding(**kwargs)

        print(f"[model_cache] Ready: {model_name}")
    return _embed_models[model_name]
