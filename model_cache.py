"""
Process-level model cache.

HuggingFace embedding models take 15-40s to load from disk on first use.
Caching the instance here means it's loaded ONCE per Flask process and
reused across all subsequent requests — making repeat queries nearly instant.
"""

_hf_models: dict = {}


def get_hf_embed(model_name: str):
    """Return a cached HuggingFaceEmbedding, loading it only on first call."""
    if model_name not in _hf_models:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"[model_cache] Loading HuggingFace embed model: {model_name} …")
        _hf_models[model_name] = HuggingFaceEmbedding(model_name=model_name)
        print(f"[model_cache] Loaded: {model_name}")
    return _hf_models[model_name]
