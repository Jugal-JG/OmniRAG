"""Shared, per-file vector indexes used by every text retrieval engine."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle

import index_cache

logger = logging.getLogger(__name__)

_CACHE_NAME = "shared_vector"


def build_or_load_file_index(filename: str, upload_dir, embed_model):
    """Load the one persisted vector index for an unchanged uploaded file."""
    file_paths = [str(upload_dir / filename)]
    cache_path = index_cache.get_cache_path(file_paths, _CACHE_NAME)
    require_meta = filename.lower().endswith(".pdf")

    def load():
        logger.info("[shared_vector] Loading cached index for %s", filename)
        context = StorageContext.from_defaults(persist_dir=str(cache_path))
        return load_index_from_storage(context)

    if index_cache.is_cache_usable(file_paths, _CACHE_NAME, require_meta=require_meta):
        return load()

    # This lock/cache key is shared by Basic RAG, Router, ReAct, Sub-Question,
    # and Multi-Document, so a waiting engine can never embed the file again.
    with index_cache.build_lock(file_paths, _CACHE_NAME):
        if index_cache.is_cache_usable(file_paths, _CACHE_NAME, require_meta=require_meta):
            return load()
        logger.info("[shared_vector] Building index for %s", filename)
        from doc_loader import load_documents

        documents = load_documents(upload_dir / filename)
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir=str(cache_path))
        index_cache.save_documents_meta(file_paths, _CACHE_NAME, documents)
        return index


class _CombinedRetriever(BaseRetriever):
    def __init__(self, indexes, similarity_top_k: int):
        self._retrievers = [
            index.as_retriever(similarity_top_k=similarity_top_k) for index in indexes
        ]
        self._top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        best = {}
        for retriever in self._retrievers:
            for result in retriever.retrieve(query_bundle):
                node_id = result.node.node_id
                existing = best.get(node_id)
                if existing is None or (result.score or 0) > (existing.score or 0):
                    best[node_id] = result
        return sorted(best.values(), key=lambda item: item.score or 0, reverse=True)[: self._top_k]


class CombinedVectorIndex:
    """VectorStoreIndex-compatible read-only view over per-file indexes."""

    def __init__(self, indexes):
        self._indexes = indexes
        docs = {}
        for index in indexes:
            docs.update(index.docstore.docs)
        self.docstore = SimpleNamespace(docs=docs)

    def as_retriever(self, similarity_top_k=1, **_kwargs):
        return _CombinedRetriever(self._indexes, similarity_top_k)


def build_or_load_indexes(filenames: list[str], upload_dir, embed_model):
    """Build per-file indexes sequentially and expose one logical index."""
    indexes = [build_or_load_file_index(filename, upload_dir, embed_model) for filename in filenames]
    return indexes[0] if len(indexes) == 1 else CombinedVectorIndex(indexes)
