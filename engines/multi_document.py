"""
Multi-Document Agent engine — mirrors multi_document_agents-v1.ipynb.

Dual-LLM architecture:
  • Top Agent   → Google Gemini 2.5 Flash (orchestration — picks which doc agents to call)
  • Doc Agents  → Google Gemini 2.5 Flash (per-doc FunctionAgents with vector + summary tools)
  • Answering   → Groq / Llama 4 Scout (fast synthesis from retrieved chunks)
  • Embeddings  → Mistral (mistral-embed)

Each uploaded file gets its own FunctionAgent (with vector + summary query engine tools).
A top-level FunctionAgent reasons across all per-document agents.
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
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from llama_index.embeddings.mistralai import MistralAIEmbedding
# Gemini Flash / Gemma orchestration and fallback synthesis
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike           # Groq — fast answering from chunks

import index_cache
from config import Config
from utils import with_retry

# FIX: Import FunctionAgent from the correct module in >=0.12
try:
    from llama_index.core.agent.workflow import FunctionAgent
except ImportError:
    from llama_index.core.agent import FunctionAgent


def _run_async(coro):
    """Run an async coroutine in a **dedicated thread** with a fresh event loop.

    nest_asyncio's patched loop does not properly propagate asyncio.current_task()
    on Python 3.14, so aiohttp's internal asyncio.timeout() raises
    ``RuntimeError: Timeout should be used inside a task``.

    By running in a brand-new thread via asyncio.run() we get a clean,
    un-patched event loop where Tasks and timeouts work correctly.
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


def _is_rate_limit(exc: Exception) -> bool:
    """Return True for provider quota/rate-limit errors."""
    exc_str = str(exc).lower()
    return (
        "429" in exc_str
        or "rate limit" in exc_str
        or "too many requests" in exc_str
        or "resource_exhausted" in exc_str
        or "quota" in exc_str
    )


class FallbackQueryEngine:
    """Query primary engine first, then fallback when provider quota is hit."""

    def __init__(self, primary_engine, fallback_engine, label: str):
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.label = label

    def query(self, query):
        try:
            return self.primary_engine.query(query)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            print(
                f"[multi-doc] {self.label}: Llama 4 Scout rate-limited; "
                f"falling back to {Config.GOOGLE_LLM}"
            )
            return self.fallback_engine.query(query)

    async def aquery(self, query):
        try:
            return await self.primary_engine.aquery(query)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            print(
                f"[multi-doc] {self.label}: Llama 4 Scout rate-limited; "
                f"falling back to {Config.GOOGLE_LLM}"
            )
            return await self.fallback_engine.aquery(query)


async def _invoke_agent(agent: FunctionAgent, query: str, max_retries: int = 5) -> str:
    """
    Invoke a FunctionAgent with automatic retry on transient server errors.

    Google's API occasionally returns 500/502/503 or MALFORMED_RESPONSE
    on the first attempts but succeeds on retry.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            handler = agent.run(query)
            result = await handler
            return str(result)
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1), 16)  # 2s, 4s, 8s, 16s, 16s
                print(f"[retry] Transient error (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait}s: {type(exc).__name__}: {str(exc)[:120]}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_exc  # should not reach here, but safety net


def _build_doc_agent(
    fname: str,
    upload_dir: Path,
    agent_llm,
    answer_llm,
    fallback_answer_llm,
    embed_model,
) -> FunctionAgent:
    """Build a per-document FunctionAgent with vector + summary query engine tools.

    - agent_llm: Gemini Flash — handles the FunctionAgent tool-calling decisions
    - answer_llm: Groq / Llama 4 Scout — fast synthesis from retrieved chunks
    - fallback_answer_llm: Gemma fallback when Groq is rate-limited
    """
    file_paths = [fname]
    vec_cache = index_cache.get_cache_path(file_paths, f"multidoc_vec_{fname}")
    sum_cache = index_cache.get_cache_path(file_paths, f"multidoc_sum_{fname}")

    from doc_loader import load_documents
    docs = load_documents(upload_dir / fname)

    # ALWAYS rebuild indices for PDFs — prevents stale empty cache from
    # poisoning results forever.  For non-PDFs, use cache normally.
    is_pdf = fname.lower().endswith(".pdf")
    vec_cached = index_cache.is_cached(file_paths, f"multidoc_vec_{fname}")
    sum_cached = index_cache.is_cached(file_paths, f"multidoc_sum_{fname}")

    if vec_cached and not is_pdf:
        vec_ctx = StorageContext.from_defaults(persist_dir=str(vec_cache))
        vector_index = load_index_from_storage(vec_ctx)
    else:
        if vec_cached:
            print(f"[multi-doc] Rebuilding vector index for '{fname}'")
        vector_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        vector_index.storage_context.persist(persist_dir=str(vec_cache))

    if sum_cached and not is_pdf:
        sum_ctx = StorageContext.from_defaults(persist_dir=str(sum_cache))
        summary_index = load_index_from_storage(sum_ctx)
    else:
        if sum_cached:
            print(f"[multi-doc] Rebuilding summary index for '{fname}'")
        summary_index = SummaryIndex.from_documents(docs)
        summary_index.storage_context.persist(persist_dir=str(sum_cache))

    stem = Path(fname).stem.replace(" ", "_")

    primary_vector_engine = vector_index.as_query_engine(
        similarity_top_k=Config.SIMILARITY_TOP_K,
        llm=answer_llm,
    )
    fallback_vector_engine = vector_index.as_query_engine(
        similarity_top_k=Config.SIMILARITY_TOP_K,
        llm=fallback_answer_llm,
    )
    primary_summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        llm=answer_llm,
    )
    fallback_summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        llm=fallback_answer_llm,
    )

    # Query engines use Groq first for speed, then Gemma if Groq hits quota.
    tools = [
        QueryEngineTool(
            query_engine=FallbackQueryEngine(
                primary_engine=primary_vector_engine,
                fallback_engine=fallback_vector_engine,
                label=f"{fname} vector search",
            ),
            metadata=ToolMetadata(
                name=f"{stem}_vector",
                description=f"Retrieves specific facts from {fname}.",
            ),
        ),
        QueryEngineTool(
            query_engine=FallbackQueryEngine(
                primary_engine=primary_summary_engine,
                fallback_engine=fallback_summary_engine,
                label=f"{fname} summary",
            ),
            metadata=ToolMetadata(
                name=f"{stem}_summary",
                description=f"Summarizes or gives overviews about {fname}.",
            ),
        ),
    ]

    # Doc agent uses Gemini Flash (agent_llm) for tool-calling decisions
    return FunctionAgent(
        tools=tools,
        llm=agent_llm,
        system_prompt=f"You are a specialized agent for answering questions about '{fname}'.",
        verbose=True,
    )


def _wrap_agent_as_tool(agent: FunctionAgent, name: str, description: str) -> FunctionTool:
    """
    Wrap an async FunctionAgent as a FunctionTool so the top-level agent
    can call it. The notebook does the same with get_agent_tool_callable().
    """
    async def query_agent(query: str) -> str:
        return await _invoke_agent(agent, query)

    return FunctionTool.from_defaults(
        async_fn=query_agent,
        name=name,
        description=description,
    )


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    # ── Dual-LLM setup ───────────────────────────────────────────────────
    # Agent LLM (Gemma 4-31b): handles orchestration + doc agent tool-calling.
    # Transient 500s are handled by _invoke_agent retry logic.
    agent_llm = GoogleGenAI(
        api_key=Config.GOOGLE_API_KEY_GEMMA,
        model=Config.GOOGLE_LLM,              # gemma-4-31b-it
        max_retries=5,
    )

    # Answering LLM (Groq / Llama 4 Scout): fast synthesis from retrieved chunks.
    # No function calling needed — just reads chunks and produces answers.
    answer_llm = OpenAILike(
        api_key=Config.GROQ_API_KEY,
        api_base="https://api.groq.com/openai/v1",
        model=Config.GROQ_LLM,
        is_chat_model=True,
        is_function_calling_model=False,
    )

    embed_model = MistralAIEmbedding(
        api_key=Config.MISTRAL_API_KEY, model_name=Config.MISTRAL_EMBED_MODEL
    )
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    all_tools = []
    thinking_steps = []

    for fname in filenames:
        thinking_steps.append(f"Building agent for: {fname}")
        agent = _build_doc_agent(
            fname,
            upload_dir,
            agent_llm,
            answer_llm,
            agent_llm,
            embed_model,
        )
        stem = Path(fname).stem.replace(" ", "_")
        all_tools.append(
            _wrap_agent_as_tool(
                agent=agent,
                name=stem,
                description=f"Agent specialized in the document: {fname}",
            )
        )

    # Top-level agent — Gemini Flash orchestrates across per-doc agents
    top_agent = FunctionAgent(
        tools=all_tools,
        llm=agent_llm,
        system_prompt=(
            "You are a top-level assistant that coordinates specialized document agents. "
            "You have access to tools — one for each uploaded document. "
            "You MUST use these tools to answer the user's question. "
            "NEVER say you don't have access. Always call the relevant document tool(s) first, "
            "then synthesize the results into a comprehensive answer."
        ),
        verbose=True,
    )

    # Debug logging
    print(f"[multi-doc] Agent LLM: {type(agent_llm).__name__} ({Config.GOOGLE_LLM})")
    print(f"[multi-doc] Answer LLM: {type(answer_llm).__name__} ({Config.GROQ_LLM})")
    print(f"[multi-doc] Tools: {[t.metadata.name for t in all_tools]}")

    # Capture verbose output as thinking steps
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = _run_async(_invoke_agent(top_agent, query))

    raw = buf.getvalue()
    if raw.strip():
        thinking_steps += [line for line in raw.splitlines() if line.strip()]

    return {
        "answer": answer,
        "sources": [],
        "thinking_steps": thinking_steps,
    }
