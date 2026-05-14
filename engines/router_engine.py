"""
Router Query Engine — mirrors Router_Query_Engine.ipynb.
LLM: Google Gemma 4-31b  |  Embeddings: HuggingFace BAAI/bge-base-en-v1.5
Internally routes between SummaryIndex and VectorStoreIndex.
"""

from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.google_genai import GoogleGenAI

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, with_retry


def _build_or_load_indexes(file_paths: list[str], upload_dir: Path):
    vec_cache = index_cache.get_cache_path(file_paths, "router_engine_vec")
    sum_cache = index_cache.get_cache_path(file_paths, "router_engine_sum")

    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    from doc_loader import load_documents
    all_docs = []
    for f in file_paths:
        all_docs.extend(load_documents(upload_dir / f))

    has_pdf = any(f.lower().endswith(".pdf") for f in file_paths)

    if not has_pdf and index_cache.is_cached(file_paths, "router_engine_vec"):
        vec_ctx = StorageContext.from_defaults(persist_dir=str(vec_cache))
        vector_index = load_index_from_storage(vec_ctx)
    else:
        vector_index = VectorStoreIndex.from_documents(all_docs)
        vector_index.storage_context.persist(persist_dir=str(vec_cache))

    if not has_pdf and index_cache.is_cached(file_paths, "router_engine_sum"):
        sum_ctx = StorageContext.from_defaults(persist_dir=str(sum_cache))
        summary_index = load_index_from_storage(sum_ctx)
    else:
        summary_index = SummaryIndex.from_documents(all_docs)
        summary_index.storage_context.persist(persist_dir=str(sum_cache))

    return vector_index, summary_index


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    llm = GoogleGenAI(api_key=Config.GOOGLE_API_KEY_GEMMA, model=Config.GOOGLE_LLM, max_retries=5)
    Settings.llm = llm

    vector_index, summary_index = _build_or_load_indexes(filenames, upload_dir)

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(similarity_top_k=Config.SIMILARITY_TOP_K),
        description="Useful for specific questions that require retrieving exact facts or passages.",
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
        description="Useful for summarization, overviews, or questions about the overall content.",
    )

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )

    response = router.query(query)
    selected = getattr(response, "metadata", {}).get("selected_tool", "")

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [f"Internal router selected: {selected}"] if selected else [],
    }
