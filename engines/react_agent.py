"""
ReAct Agent engine — mirrors ReAct_Agent.ipynb.
LLM: Google Gemini 2.5 Flash via GOOGLE_API_KEY (primary) / Gemini 3.1 Flash Lite via GOOGLE_API_KEY2 (fallback)  |  Embeddings: HuggingFace BAAI/bge-m3 (was: microsoft/harrier-oss-v1-270m)
Uses step-by-step tool reasoning; captures verbose output as thinking steps.
"""

import asyncio
import io
import contextlib
import concurrent.futures
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI

import model_cache
import shared_vector_index
import retrieval
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
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    return shared_vector_index.build_or_load_file_index(fname, upload_dir, embed_model)


def _make_llm(model: str, api_key: str | None = None) -> GoogleGenAI:
    return GoogleGenAI(
        api_key=api_key or Config.GOOGLE_API_KEY,
        model=model,
        max_retries=Config.GOOGLE_MAX_RETRIES,
        temperature=0,
    )


def _model_unavailable(exc: Exception) -> bool:
    """A hard error meaning THIS model can't serve — worth trying the fallback."""
    s = str(exc).lower()
    return any(
        k in s
        for k in ("429", "resource_exhausted", "quota", "404", "not found",
                  "unsupported", "permission", "unavailable", "503")
    )


_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the uploaded documents.\n"
    "You have NO prior knowledge of the documents' contents. You do not know any\n"
    "formulas, equations, numbers, or facts from them until you retrieve them.\n"
    "Therefore, for EVERY question, your first step MUST be to call a document\n"
    "search tool (emit an `Action:`). Never answer a factual question from your own\n"
    "knowledge — that would be a hallucination.\n\n"
    "The search tool returns VERBATIM excerpts from the document. Read them and base\n"
    "your answer only on their content. Equations, formulas, tables and figures are\n"
    "labelled with a parenthesised number such as (15): a question about 'formula 15',\n"
    "'equation 15', or 'eq. 15' refers to the expression labelled (15) in the\n"
    "excerpts — locate that label and report the corresponding expression. Only say\n"
    "you could not find it if no such label or content appears in the excerpts.\n\n"
    "MATH DELIMITERS REQUIREMENT:\n"
    "For any mathematical formulas, equations, or symbols, you MUST format them in "
    "LaTeX using these delimiters STRICTLY:\n"
    "- Use $$...$$ on its OWN LINE for ALL block equations, matrices, multi-component "
    "formulas, or any formula that spans more than one symbol. Example:\n"
    "  $$\\mathbf{P}_{i,j} = \\begin{pmatrix} P^{xx} & P^{xv} \\\\ P^{vx} & P^{vv} \\end{pmatrix}$$\n"
    "- Use $...$ ONLY for single, short inline symbols like 'where $n$ is the count'.\n"
    "- NEVER use $...$ for multi-line expressions, matrices, or formulas with subscripts "
    "and superscripts that span more than a few tokens — use $$...$$ instead.\n"
    "- NEVER use plain brackets [ ] or parentheses ( ) for equations.\n"
    "- NEVER split a single formula across multiple $...$ inline spans."
)


def _make_search_tool(fname: str, retriever, sources_sink: list) -> FunctionTool:
    """A tool that returns raw retrieved excerpts — the agent's own LLM reads them.

    Returning verbatim text (instead of a second LLM synthesising an answer)
    removes the failure mode where the synthesiser refuses with 'not found'
    despite the relevant chunk being retrieved, and saves one LLM call per step.
    Retrieved nodes are captured in ``sources_sink`` so the UI can still show them.
    """

    def search_document(query: str) -> str:
        nodes = retriever.retrieve(query)
        sources_sink.extend(nodes)
        return retrieval.format_context(nodes)

    return FunctionTool.from_defaults(
        fn=search_document,
        name=Path(fname).stem.replace(" ", "_").replace("-", "_"),
        description=(
            f"Search the document '{fname}' and return the most relevant verbatim "
            "excerpts. Input: a search query string. Always call this before answering."
        ),
    )


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP

    # Tools no longer depend on the LLM (raw retrieval), so build them once.
    collected_sources: list = []
    tools = []
    for fname in filenames:
        idx = _build_or_load_index(fname, upload_dir)
        retriever = retrieval.make_retriever(idx, similarity_top_k=Config.SIMILARITY_TOP_K)
        tools.append(_make_search_tool(fname, retriever, collected_sources))

    def _run_with(model: str, api_key: str | None = None):
        llm = _make_llm(model, api_key=api_key)
        Settings.llm = llm
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            system_prompt=_SYSTEM_PROMPT,
            verbose=True,
            max_iterations=10,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resp = _run_async(_invoke_agent(agent, query))
        return resp, buf.getvalue()

    primary_model = Config.REACT_PRIMARY_LLM          # gemini-2.5-flash
    fallback_model = Config.GOOGLE_LLM                 # gemini-3.1-flash-lite-preview
    fallback_key = Config.GOOGLE_API_KEY2 or None

    # Attempt 1: primary key (GOOGLE_API_KEY) + primary model (gemini-2.5-flash)
    try:
        response, raw_steps = _run_with(primary_model)
    except Exception as exc1:
        if not _model_unavailable(exc1):
            raise
        print(f"[react] GOOGLE_API_KEY + '{primary_model}' unavailable "
              f"({type(exc1).__name__}: {str(exc1)[:120]})")

        # Attempt 2: fallback key (GOOGLE_API_KEY2) + fallback model (gemini-3.1-flash-lite-preview)
        if fallback_key:
            print(f"[react] retrying with GOOGLE_API_KEY2 + '{fallback_model}'")
            response, raw_steps = _run_with(fallback_model, api_key=fallback_key)
        else:
            # No second key configured — re-raise the original error
            print("[react] no GOOGLE_API_KEY2 set; no further fallback available")
            raise

    thinking_steps = [line for line in raw_steps.splitlines() if line.strip()]

    answer = str(response)
    # Prefer nodes captured from the tool calls; fall back to any on the response.
    source_nodes = collected_sources or getattr(response, "source_nodes", [])

    return {
        "answer": answer,
        "sources": format_source_nodes(source_nodes),
        "thinking_steps": thinking_steps,
    }
