"""
Basic RAG engine — mirrors Basic_RAG_With_LlamaIndex.ipynb.
LLM: Mistral Large  |  Embeddings: HuggingFace BAAI/bge-base-en-v1.5
"""

from pathlib import Path
import logging

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.mistralai import MistralAI

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, with_retry

logger = logging.getLogger(__name__)


def _build_or_load_index(file_paths: list[str], upload_dir: Path):
    cache_file_paths = [str(upload_dir / f) for f in file_paths]
    cache_path = index_cache.get_cache_path(cache_file_paths, "basic_rag")

    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    has_pdf = any(f.lower().endswith(".pdf") for f in file_paths)

    if index_cache.is_cache_usable(cache_file_paths, "basic_rag", require_meta=has_pdf):
        logger.info("[basic_rag] Loading cached vector index for %s", file_paths)
        storage_ctx = StorageContext.from_defaults(persist_dir=str(cache_path))
        index = load_index_from_storage(storage_ctx)
    else:
        logger.info("[basic_rag] Building vector index for %s", file_paths)
        from doc_loader import load_documents
        all_docs = []
        for f in file_paths:
            all_docs.extend(load_documents(upload_dir / f))
        index = VectorStoreIndex.from_documents(all_docs)
        index.storage_context.persist(persist_dir=str(cache_path))
        index_cache.save_documents_meta(cache_file_paths, "basic_rag", all_docs)

    return index


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    llm = MistralAI(api_key=Config.MISTRAL_API_KEY, model=Config.MISTRAL_LLM)
    Settings.llm = llm
    logger.info("[basic_rag] LLM=MistralAI (%s)", Config.MISTRAL_LLM)

    index = _build_or_load_index(filenames, upload_dir)
    engine = index.as_query_engine(similarity_top_k=Config.SIMILARITY_TOP_K)
    response = engine.query(query)

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [],
    }
