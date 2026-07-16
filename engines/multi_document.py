"""Multi-document agent engine.

The engine still follows the notebook architecture: each uploaded file gets its
own FunctionAgent, and a top-level FunctionAgent coordinates them. The document
processing path fans out to the per-document agents concurrently so multi-file
queries do not wait for one document to finish before starting the next.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    SummaryIndex,
    load_index_from_storage,
)
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

import index_cache
import model_cache
import shared_vector_index
from answer_format import MATH_FORMAT_INSTRUCTIONS
from config import Config

logger = logging.getLogger(__name__)

try:
    from llama_index.core.agent.workflow import FunctionAgent
except ImportError:
    from llama_index.core.agent import FunctionAgent


@dataclass
class DocumentAgent:
    filename: str
    agent: FunctionAgent


def _run_async(coro):
    """Run a coroutine in a dedicated thread with a fresh event loop.

    Flask's request thread may already have loop state, and Python 3.14 is
    stricter about asyncio timeouts needing a real Task context. A short-lived
    worker thread plus asyncio.run gives each agent run clean loop ownership.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient provider error."""
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


def _should_fallback(exc: Exception) -> bool:
    return _is_rate_limit(exc) or _is_retryable(exc)


class FallbackQueryEngine:
    """Query the primary answer engine, then fall back quickly on provider errors."""

    def __init__(
        self,
        primary_engine,
        fallback_engine,
        label: str,
        primary_model: str,
        fallback_model: str,
    ):
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.label = label
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def query(self, query):
        try:
            logger.info("[multi-doc] %s: primary query using %s", self.label, self.primary_model)
            response = self.primary_engine.query(query)
            logger.info("[multi-doc] %s: primary query succeeded", self.label)
            return response
        except Exception as exc:
            if not _should_fallback(exc):
                raise
            logger.info(
                "[multi-doc] %s: primary answer model failed (%s); falling back to %s",
                self.label,
                type(exc).__name__,
                self.fallback_model,
            )
            response = self.fallback_engine.query(query)
            logger.info("[multi-doc] %s: fallback query succeeded", self.label)
            return response

    async def aquery(self, query):
        try:
            logger.info(
                "[multi-doc] %s: primary async query using %s",
                self.label,
                self.primary_model,
            )
            response = await self.primary_engine.aquery(query)
            logger.info("[multi-doc] %s: primary async query succeeded", self.label)
            return response
        except Exception as exc:
            if not _should_fallback(exc):
                raise
            logger.info(
                "[multi-doc] %s: primary answer model failed (%s); falling back to %s",
                self.label,
                type(exc).__name__,
                self.fallback_model,
            )
            response = await self.fallback_engine.aquery(query)
            logger.info("[multi-doc] %s: fallback async query succeeded", self.label)
            return response


def _agent_result_text(result) -> str:
    """Extract user-visible text from LlamaIndex agent outputs."""
    response = getattr(result, "response", None)
    if response is not None:
        content = getattr(response, "content", None)
        if content:
            return str(content)

    for attr in ("content", "text"):
        value = getattr(result, attr, None)
        if value:
            return str(value)

    return str(result)


async def _invoke_agent(agent: FunctionAgent, query: str, max_retries: int | None = None) -> str:
    """Invoke a FunctionAgent with a small retry budget for transient Google errors."""
    attempts = max_retries or Config.GOOGLE_MAX_RETRIES
    last_exc = None
    for attempt in range(attempts):
        try:
            handler = agent.run(query)
            result = await handler
            return _agent_result_text(result)
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < attempts - 1:
                wait = min(2 ** attempt, 4)
                logger.info(
                    "[retry] Transient agent error (attempt %s/%s), retrying in %ss: %s: %s",
                    attempt + 1,
                    attempts,
                    wait,
                    type(exc).__name__,
                    str(exc)[:120],
                )
                await asyncio.sleep(wait)
                continue
            raise
    raise last_exc


def _cache_file_paths(fname: str, upload_dir: Path) -> list[str]:
    return [str(upload_dir / fname)]


def _load_or_build_indexes(fname: str, upload_dir: Path, embed_model):
    file_paths = _cache_file_paths(fname, upload_dir)
    sum_engine = f"multidoc_sum_{fname}"
    sum_cache = index_cache.get_cache_path(file_paths, sum_engine)
    is_pdf = fname.lower().endswith(".pdf")
    docs = None

    def get_docs():
        nonlocal docs
        if docs is None:
            from doc_loader import load_documents

            docs = load_documents(upload_dir / fname)
        return docs

    vector_index = shared_vector_index.build_or_load_file_index(
        fname, upload_dir, embed_model
    )

    if index_cache.is_cache_usable(file_paths, sum_engine, require_meta=is_pdf):
        sum_ctx = StorageContext.from_defaults(persist_dir=str(sum_cache))
        summary_index = load_index_from_storage(sum_ctx)
    else:
        loaded_docs = get_docs()
        summary_index = SummaryIndex.from_documents(loaded_docs)
        summary_index.storage_context.persist(persist_dir=str(sum_cache))
        index_cache.save_documents_meta(file_paths, sum_engine, loaded_docs)

    return vector_index, summary_index


def _build_doc_agent(
    fname: str,
    upload_dir: Path,
    agent_llm,
    answer_llm,
    fallback_answer_llm,
    embed_model,
) -> DocumentAgent:
    """Build a per-document FunctionAgent with vector and summary tools."""
    vector_index, summary_index = _load_or_build_indexes(fname, upload_dir, embed_model)
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

    tools = [
        QueryEngineTool(
            query_engine=FallbackQueryEngine(
                primary_engine=primary_vector_engine,
                fallback_engine=fallback_vector_engine,
                label=f"{fname} vector search",
                primary_model=Config.GOOGLE_LLM,
                fallback_model=Config.GROQ_LLM,
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
                primary_model=Config.GOOGLE_LLM,
                fallback_model=Config.GROQ_LLM,
            ),
            metadata=ToolMetadata(
                name=f"{stem}_summary",
                description=f"Summarizes or gives overviews about {fname}.",
            ),
        ),
    ]

    agent = FunctionAgent(
        name=stem,
        description=f"Answers questions about {fname}.",
        tools=tools,
        llm=agent_llm,
        system_prompt=(
            f"You are a specialized document agent for '{fname}'. "
            "Use the provided vector and summary tools before answering. "
            "Answer only from this document. For cross-document comparison requests, "
            "do not compare and do not refuse; summarize the facts, themes, and "
            "details from this one document that a coordinator can compare later."
            + MATH_FORMAT_INSTRUCTIONS
        ),
        verbose=False,
        allow_parallel_tool_calls=True,
    )
    return DocumentAgent(filename=fname, agent=agent)


async def _query_document_agents(doc_agents: list[DocumentAgent], query: str) -> list[dict]:
    async def query_one(doc_agent: DocumentAgent) -> dict:
        doc_query = (
            "Single-document extraction task: use your tools to summarize this one "
            "document's main topic, key facts, themes, and details relevant to the "
            "user's eventual cross-document question. Do not compare against other "
            "documents. Do not refuse because you only see one document.\n\n"
            f"User's cross-document question: {query}\n{MATH_FORMAT_INSTRUCTIONS}"
        )
        try:
            logger.info("[multi-doc] %s: agent started", doc_agent.filename)
            answer = await _invoke_agent(doc_agent.agent, doc_query)
            logger.info(
                "[multi-doc] %s: agent completed answer_chars=%s",
                doc_agent.filename,
                len(answer),
            )
            return {"file": doc_agent.filename, "answer": answer, "error": None}
        except Exception as exc:
            logger.info(
                "[multi-doc] %s: agent failed %s: %s",
                doc_agent.filename,
                type(exc).__name__,
                str(exc)[:200],
            )
            return {
                "file": doc_agent.filename,
                "answer": "",
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            }

    return await asyncio.gather(*(query_one(doc_agent) for doc_agent in doc_agents))


def _format_doc_results(results: list[dict]) -> str:
    blocks = []
    for result in results:
        if result["error"]:
            body = f"ERROR: {result['error']}"
        else:
            body = result["answer"]
        blocks.append(f"Document: {result['file']}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _build_parallel_document_tool(
    doc_agents: list[DocumentAgent],
    result_holder: dict,
) -> FunctionTool:
    """Create the top-agent tool that queries all document agents in parallel."""

    async def query_all_documents(query: str) -> str:
        results = await _query_document_agents(doc_agents, query)
        result_holder["results"] = results
        successful = [result for result in results if result["answer"]]
        if not successful:
            errors = "; ".join(f"{r['file']}: {r['error']}" for r in results)
            return f"All document agents failed. {errors}"
        return _format_doc_results(results)

    names = ", ".join(doc_agent.filename for doc_agent in doc_agents)
    return FunctionTool.from_defaults(
        async_fn=query_all_documents,
        name="parallel_document_agents",
        description=(
            "Queries all specialized document agents concurrently and returns "
            f"their per-document answers. Documents: {names}"
        ),
    )


async def _synthesize_from_results(llm, query: str, results: list[dict]) -> str:
    successful = [result for result in results if result["answer"]]
    if not successful:
        errors = "; ".join(f"{r['file']}: {r['error']}" for r in results)
        raise RuntimeError(f"All document agents failed. {errors}")

    prompt = (
        "Synthesize the following parallel document-agent results into a clear answer. "
        "Use only these results. If they disagree, explain the difference. "
        "If one document failed or lacks the answer, say so briefly.\n\n"
        f"User question:\n{query}\n\n"
        f"Document-agent results:\n{_format_doc_results(results)}\n\n"
        "Final answer:"
        + MATH_FORMAT_INSTRUCTIONS
    )
    try:
        response = await llm.acomplete(prompt)
        text = getattr(response, "text", str(response)).strip()
        if text:
            return text
    except Exception as exc:
        logger.info("[multi-doc] Direct synthesis failed, returning document results: %s", exc)

    return _format_doc_results(successful)


def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    agent_llm = GoogleGenAI(
        api_key=Config.GOOGLE_API_KEY,
        model=Config.GOOGLE_LLM,
        max_retries=Config.GOOGLE_MAX_RETRIES,
    )

    answer_llm = agent_llm
    fallback_answer_llm = None
    if Config.GROQ_API_KEY:
        fallback_answer_llm = OpenAILike(
            api_key=Config.GROQ_API_KEY,
            api_base="https://api.groq.com/openai/v1",
            model=Config.GROQ_LLM,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=Config.GROQ_CONTEXT_WINDOW,
            max_tokens=Config.ANSWER_MAX_TOKENS,
        )

    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.llm = agent_llm
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP

    thinking_steps = [
        f"Agent model: {Config.GOOGLE_LLM}",
        f"Answer model: {Config.GOOGLE_LLM}",
        f"Fallback answer model: {Config.GROQ_LLM if fallback_answer_llm else 'none'}",
        "Execution: per-document agents queried in parallel",
    ]

    doc_agents = []
    for fname in filenames:
        thinking_steps.append(f"Preparing agent for: {fname}")
        doc_agents.append(
            _build_doc_agent(
                fname=fname,
                upload_dir=upload_dir,
                agent_llm=agent_llm,
                answer_llm=answer_llm,
                fallback_answer_llm=fallback_answer_llm or agent_llm,
                embed_model=embed_model,
            )
        )

    logger.info("[multi-doc] Agent LLM: %s (%s)", type(agent_llm).__name__, Config.GOOGLE_LLM)
    logger.info(
        "[multi-doc] Answer LLM: %s (%s)",
        type(answer_llm).__name__,
        Config.GOOGLE_LLM,
    )
    if fallback_answer_llm:
        logger.info(
            "[multi-doc] Fallback answer LLM: %s (%s)",
            type(fallback_answer_llm).__name__,
            Config.GROQ_LLM,
        )
    result_holder = {}
    parallel_tool = _build_parallel_document_tool(doc_agents, result_holder)
    top_agent = FunctionAgent(
        name="multi_document_coordinator",
        description="Coordinates specialized document agents for cross-document QA.",
        tools=[parallel_tool],
        llm=agent_llm,
        system_prompt=(
            "You are a top-level assistant that coordinates specialized document agents. "
            "You MUST call the parallel_document_agents tool before answering. "
            "Use only the tool results to answer the user. If document agents disagree, "
            "explain the difference. If a document failed or lacks the answer, say so briefly."
            + MATH_FORMAT_INSTRUCTIONS
        ),
        verbose=False,
        allow_parallel_tool_calls=True,
    )

    logger.info("[multi-doc] Parallel agents: %s", [doc_agent.filename for doc_agent in doc_agents])

    answer = _run_async(_invoke_agent(top_agent, query))
    if not answer.strip() and result_holder.get("results"):
        logger.info("[multi-doc] Coordinator returned empty text; using direct synthesis fallback")
        answer = _run_async(
            _synthesize_from_results(agent_llm, query, result_holder["results"])
        )

    for result in result_holder.get("results", []):
        status = "failed" if result["error"] else "answered"
        thinking_steps.append(f"{result['file']}: {status}")

    return {
        "answer": answer,
        "sources": [],
        "thinking_steps": thinking_steps,
    }
