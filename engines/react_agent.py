"""
ReAct Agent engine — mirrors ReAct_Agent.ipynb.
LLM: Google Gemini 2.5 Flash  |  Embeddings: HuggingFace BAAI/bge-base-en-v1.5
Uses step-by-step tool reasoning; captures verbose output as thinking steps.
"""

import asyncio
import io
import contextlib
import concurrent.futures
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.google_genai import GoogleGenAI

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, with_retry


def _run_async(coro):
    """Run an async coroutine in a dedicated thread with a fresh event loop.

    Bypasses nest_asyncio issues on Python 3.14 where asyncio.timeout()
    (used by aiohttp / google-genai) requires a proper Task context.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient error worth retrying."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    return (
        "500" in exc_str
        or "502" in exc_str
        or "503" in exc_str
        or "servererror" in exc_type
        or "internal" in exc_str
        or "malformed_response" in exc_str
        or "terminated early" in exc_str
    )


async def _invoke_agent(agent: ReActAgent, query: str, max_retries: int = 5):
    """
    Invoke a ReActAgent with automatic retry on transient server errors.

    Google's API occasionally returns 500/502/503 or MALFORMED_RESPONSE
    on the first attempts but succeeds on retry.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            handler = agent.run(query)
            result = await handler
            return result
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1), 16)
                print(f"[retry] Transient error (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait}s: {type(exc).__name__}: {str(exc)[:120]}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_exc

def _build_or_load_index(fname: str, upload_dir: Path):
    file_paths = [fname]
    cache_path = index_cache.get_cache_path(file_paths, f"react_{fname}")
    is_pdf = fname.lower().endswith(".pdf")

    # For non-PDFs, use cache if available
    if not is_pdf and index_cache.is_cached(file_paths, f"react_{fname}"):
        ctx = StorageContext.from_defaults(persist_dir=str(cache_path))
        return load_index_from_storage(ctx)

    # Always re-parse PDFs (cache may have been built with empty content)
    from doc_loader import load_documents
    docs = load_documents(upload_dir / fname)

    if is_pdf:
        print(f"[react] Rebuilding index for '{fname}'")
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(cache_path))
    return index


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    llm = GoogleGenAI(api_key=Config.GOOGLE_API_KEY, model=Config.GEMINI_LLM, max_retries=5)
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    tools = []
    for fname in filenames:
        idx = _build_or_load_index(fname, upload_dir)
        qe = idx.as_query_engine(similarity_top_k=Config.SIMILARITY_TOP_K)
        tools.append(
            QueryEngineTool(
                query_engine=qe,
                metadata=ToolMetadata(
                    name=Path(fname).stem.replace(" ", "_"),
                    description=f"Provides information from document: {fname}",
                ),
            )
        )

    # agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, max_iterations=10)
    agent = ReActAgent(tools=tools, llm=llm, verbose=True, max_iterations=10)


    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        response = _run_async(_invoke_agent(agent, query))


    raw_steps = buf.getvalue()
    thinking_steps = [line for line in raw_steps.splitlines() if line.strip()]

    answer = str(response)
    source_nodes = getattr(response, "source_nodes", [])


    return {
        # "answer": str(response),
        # "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "answer": answer,
        "sources": format_source_nodes(source_nodes),
        "thinking_steps": thinking_steps,
    }
