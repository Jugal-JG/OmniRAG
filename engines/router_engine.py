"""Router Query Engine.

Internally routes between SummaryIndex and VectorStoreIndex.
LLM: Groq Qwen, with Gemini Flash-Lite fallback on Groq rate limits.
"""

from pathlib import Path
import logging
import re
import time

from llama_index.core import (
    Settings,
    StorageContext,
    SummaryIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

import index_cache
import model_cache
import shared_vector_index
from config import Config
from utils import format_source_nodes, is_daily_quota_error, with_retry

logger = logging.getLogger(__name__)

ANSWER_FORMAT_INSTRUCTIONS = """

Answer formatting:
- Return only the final answer for the user. Never reveal analysis, planning,
  reasoning, self-checks, draft text, constraints, or labels such as "Output".
- Start directly with the answer; do not preface it with "Here is my thinking".
- Write currency as `USD 59,696`, not `$...$`, so it is never mistaken for math.
- Use clean Markdown.
- If the answer discusses more than one person, company, document, or topic, use one short heading per item and bullet points under each.
- If the user asks for impact, comparison, pros/cons, causes, or details, prefer bullets over one long paragraph.
- Keep very short factual answers to one concise paragraph.
"""


def _requested_word_limit(query: str) -> int | None:
    """Return a user-requested maximum word count, when one is explicit."""
    match = re.search(
        r"\b(?:under|below|less\s+than|within|maximum\s+of|at\s+most|max)\s+(\d+)\s+words?\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    requested = int(match.group(1))
    # "Less than 200 words" means at most 199 words.
    if re.search(r"\bless\s+than\b", match.group(0), flags=re.IGNORECASE):
        requested -= 1
    return max(1, requested)


def _clean_response(text: str, word_limit: int | None) -> str:
    """Remove accidental model scratchpad text and enforce an explicit word limit."""
    answer = str(text or "").strip()

    # Some reasoning models return their internal draft followed by
    # "[Output] -> final answer" in the same content field. Keep only the
    # final output section. Normal answers are left unchanged.
    markers = list(re.finditer(r"(?im)^\s*\[(?:final\s+)?output(?:\s+generation)?\]\s*(?:->|:)?\s*", answer))
    if markers:
        answer = answer[markers[-1].end():]
        answer = re.split(
            r"(?im)^\s*(?:✅|all constraints|self-correction|check against constraints|output matches|final check|proceeds\b)",
            answer,
            maxsplit=1,
        )[0]
        answer = answer.strip(' \n\t"')

    if word_limit is not None:
        words = re.findall(r"\S+", answer)
        if len(words) > word_limit:
            answer = " ".join(words[:word_limit]).rstrip(".,;:") + "…"
    return answer


def _make_groq_llm():
    return OpenAILike(
        api_key=Config.GROQ_API_KEY,
        api_base="https://api.groq.com/openai/v1",
        model=Config.GROQ_ROUTER_LLM,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=Config.GROQ_CONTEXT_WINDOW,
        temperature=0,
        max_tokens=Config.ANSWER_MAX_TOKENS,
        max_retries=0,
        # Qwen 3.6 is a reasoning model. Without this Groq can place its
        # scratchpad directly in the visible answer content.
        # The bundled OpenAI-compatible client supports reasoning_effort but
        # not Groq's newer reasoning_format parameter.
        additional_kwargs={"reasoning_effort": "none"},
    )


def _make_gemini_llm():
    return GoogleGenAI(
        api_key=Config.GOOGLE_API_KEY,
        model=Config.GOOGLE_LLM,
        temperature=0,
        max_tokens=Config.ANSWER_MAX_TOKENS,
        max_retries=Config.GOOGLE_MAX_RETRIES,
        is_function_calling_model=False,
    )


def _build_or_load_indexes(file_paths: list[str], upload_dir: Path):
    cache_file_paths = [str(upload_dir / f) for f in file_paths]
    sum_cache = index_cache.get_cache_path(cache_file_paths, "router_engine_sum")

    embed_model = model_cache.get_embed_model(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP

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

    vector_index = shared_vector_index.build_or_load_indexes(
        file_paths, upload_dir, embed_model
    )

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
    import retrieval

    Settings.llm = llm
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retrieval.make_query_engine(
            vector_index,
            similarity_top_k=Config.SIMILARITY_TOP_K,
            llm=llm,
        ),
        description=(
            "Choose this for targeted retrieval: exact facts, calculations, names, "
            "tables, sections, or labelled equations/formulas, including questions "
            "about several specific items."
        ),
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize",
            llm=llm,
        ),
        description=(
            "Choose this only for broad whole-document summaries, overviews, themes, "
            "or main ideas. Do not choose it for exact values or labelled document items."
        ),
    )
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=False,
    )


def _formatted_query(query: str) -> str:
    from answer_format import MATH_FORMAT_INSTRUCTIONS
    word_limit = _requested_word_limit(query)
    limit_instruction = (
        f"\n- Your final answer must contain no more than {word_limit} words."
        if word_limit is not None else ""
    )
    return f"{query.strip()}\n{ANSWER_FORMAT_INSTRUCTIONS}{limit_instruction}\n{MATH_FORMAT_INSTRUCTIONS}"


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    start = time.perf_counter()
    word_limit = _requested_word_limit(query)
    llm = _make_groq_llm()
    Settings.llm = llm
    logger.info("[router_engine] LLM=OpenAILike/Groq (%s)", Config.GROQ_ROUTER_LLM)

    # Broad workbook questions can use the compact sheet profiles immediately,
    # without waiting for the background vector job.
    from spreadsheet_store import profile_context

    structured_profile = profile_context(filenames, upload_dir)
    if structured_profile:
        logger.info("[router_engine] using SQLite workbook profile; vector index not required")
        prompt = (
            "Answer the question only from these verified workbook profiles. "
            "State when the profile is insufficient for a detailed claim.\n\n"
            f"{structured_profile}\n\nQuestion: {_formatted_query(query)}"
        )
        try:
            response = llm.complete(prompt)
        except Exception as exc:
            from utils import is_rate_limit_error
            if not is_rate_limit_error(exc):
                raise
            logger.info("[router_engine] Groq rate limit; falling back to Gemini %s", Config.GOOGLE_LLM)
            response = _make_gemini_llm().complete(prompt)
        return {
            "answer": _clean_response(str(response), word_limit),
            "sources": [
                {"file": filename, "text": "Structured workbook profile", "score": 1.0}
                for filename in filenames
            ],
            "thinking_steps": ["Answered from local workbook profiles while semantic indexing continues in the background."],
        }

    vector_index, summary_index = _build_or_load_indexes(filenames, upload_dir)
    logger.info("[router_engine] index setup took %.2fs", time.perf_counter() - start)

    query_start = time.perf_counter()
    try:
        response = _make_router(vector_index, summary_index, llm).query(
            _formatted_query(query)
        )
    except Exception as exc:
        from utils import is_rate_limit_error
        if not is_rate_limit_error(exc):
            raise
        logger.info("[router_engine] Groq rate limit; falling back to Gemini %s", Config.GOOGLE_LLM)
        response = _make_router(vector_index, summary_index, _make_gemini_llm()).query(
            _formatted_query(query)
        )
    selected = getattr(response, "metadata", {}).get("selected_tool", "")

    logger.info("[router_engine] query execution took %.2fs", time.perf_counter() - query_start)
    return {
        "answer": _clean_response(str(response), word_limit),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [f"Internal router selected: {selected}"] if selected else [],
    }
