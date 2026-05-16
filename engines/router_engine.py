"""Router Query Engine.

Internally routes between SummaryIndex and VectorStoreIndex.
LLM: Groq Llama 3.3, with Gemini Flash-Lite fallback.
"""

from pathlib import Path
import logging
import time

from llama_index.core import (
    Settings,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, with_retry

logger = logging.getLogger(__name__)

ANSWER_FORMAT_INSTRUCTIONS = """

Answer formatting:
- Use clean Markdown.
- If the answer discusses more than one person, company, document, or topic, use one short heading per item and bullet points under each.
- If the user asks for impact, comparison, pros/cons, causes, or details, prefer bullets over one long paragraph.
- Keep very short factual answers to one concise paragraph.
"""


def _make_groq_llm():
    return OpenAILike(
        api_key=Config.GROQ_API_KEY,
        api_base="https://api.groq.com/openai/v1",
        model=Config.GROQ_SUBQUESTION_LLM,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0,
        max_tokens=1024,
    )


def _make_gemini_llm():
    return GoogleGenAI(
        api_key=Config.GOOGLE_API_KEY,
        model=Config.GOOGLE_LLM,
        temperature=0,
        max_tokens=1024,
        max_retries=Config.GOOGLE_MAX_RETRIES,
        is_function_calling_model=False,
    )


def _build_or_load_indexes(file_paths: list[str], upload_dir: Path):
    cache_file_paths = [str(upload_dir / f) for f in file_paths]
    vec_cache = index_cache.get_cache_path(cache_file_paths, "router_engine_vec")
    sum_cache = index_cache.get_cache_path(cache_file_paths, "router_engine_sum")

    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    has_pdf = any(f.lower().endswith(".pdf") for f in file_paths)
    all_docs = None

    def get_docs():
        nonlocal all_docs
        if all_docs is None:
            from doc_loader import load_documents

            all_docs = []
            for f in file_paths:
                all_docs.extend(load_documents(upload_dir / f))
        return all_docs

    if index_cache.is_cache_usable(
        cache_file_paths, "router_engine_vec", require_meta=has_pdf
    ):
        logger.info("[router_engine] Loading cached vector index for %s", file_paths)
        vec_ctx = StorageContext.from_defaults(persist_dir=str(vec_cache))
        vector_index = load_index_from_storage(vec_ctx)
    else:
        logger.info("[router_engine] Building vector index for %s", file_paths)
        docs = get_docs()
        vector_index = VectorStoreIndex.from_documents(docs)
        vector_index.storage_context.persist(persist_dir=str(vec_cache))
        index_cache.save_documents_meta(cache_file_paths, "router_engine_vec", docs)

    if index_cache.is_cache_usable(
        cache_file_paths, "router_engine_sum", require_meta=has_pdf
    ):
        logger.info("[router_engine] Loading cached summary index for %s", file_paths)
        sum_ctx = StorageContext.from_defaults(persist_dir=str(sum_cache))
        summary_index = load_index_from_storage(sum_ctx)
    else:
        logger.info("[router_engine] Building summary index for %s", file_paths)
        docs = get_docs()
        summary_index = SummaryIndex.from_documents(docs)
        summary_index.storage_context.persist(persist_dir=str(sum_cache))
        index_cache.save_documents_meta(cache_file_paths, "router_engine_sum", docs)

    return vector_index, summary_index


def _make_router(vector_index, summary_index, llm):
    Settings.llm = llm
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            similarity_top_k=Config.SIMILARITY_TOP_K,
            llm=llm,
        ),
        description="Useful for specific questions that require retrieving exact facts or passages.",
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize",
            llm=llm,
        ),
        description="Useful for summarization, overviews, or questions about the overall content.",
    )
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=False,
    )


def _formatted_query(query: str) -> str:
    return f"{query.strip()}\n{ANSWER_FORMAT_INSTRUCTIONS}"


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    start = time.perf_counter()
    llm = _make_groq_llm()
    Settings.llm = llm
    logger.info("[router_engine] LLM=OpenAILike/Groq (%s)", Config.GROQ_SUBQUESTION_LLM)

    vector_index, summary_index = _build_or_load_indexes(filenames, upload_dir)
    logger.info("[router_engine] index setup took %.2fs", time.perf_counter() - start)

    query_start = time.perf_counter()
    try:
        response = _make_router(vector_index, summary_index, llm).query(_formatted_query(query))
    except Exception as exc:
        logger.info(
            "[router_engine] Groq run failed (%s: %s); falling back to Gemini %s",
            type(exc).__name__,
            str(exc)[:240],
            Config.GOOGLE_LLM,
        )
        response = _make_router(vector_index, summary_index, _make_gemini_llm()).query(
            _formatted_query(query)
        )

    logger.info("[router_engine] query execution took %.2fs", time.perf_counter() - query_start)
    selected = getattr(response, "metadata", {}).get("selected_tool", "")

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [f"Internal router selected: {selected}"] if selected else [],
    }
