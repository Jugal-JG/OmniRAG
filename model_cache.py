"""
Process-level model cache.

HuggingFace embedding models take 15-40s to load from disk on first use.
Caching the instance here means it's loaded ONCE per Flask process and
reused across all subsequent requests — making repeat queries nearly instant.
"""

_hf_models: dict = {}

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


def get_hf_embed(model_name: str):
    """Return a cached HuggingFaceEmbedding, loading it only on first call."""
    if model_name not in _hf_models:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"[model_cache] Loading HuggingFace embed model: {model_name} …")

        kwargs = dict(model_name=model_name, trust_remote_code=True)

        query_instr = _QUERY_INSTRUCTIONS.get(model_name)
        if query_instr:
            kwargs["query_instruction"] = query_instr
            kwargs["text_instruction"] = ""  # documents get no prefix

        _hf_models[model_name] = HuggingFaceEmbedding(**kwargs)
        print(f"[model_cache] Loaded: {model_name}")
    return _hf_models[model_name]
