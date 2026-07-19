"""
Sub-Question Query Engine — mirrors SubQuestion_Query_Engine.ipynb.
LLM: Groq Llama 3.3  |  Embeddings: HuggingFace BAAI/bge-m3 (was: microsoft/harrier-oss-v1-270m)
Decomposes complex queries into per-document sub-questions then synthesizes.
"""

from pathlib import Path
import logging
import time

from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

import async_runtime
import model_cache
import shared_vector_index
from config import Config
from utils import format_source_nodes, is_daily_quota_error, is_rate_limit_error, with_retry

logger = logging.getLogger(__name__)

ANSWER_FORMAT_INSTRUCTIONS = """

Answer formatting:
- Use clean Markdown.
- If the answer discusses more than one person, company, document, or topic, use one short heading per item and bullet points under each.
- If the user asks for impact, comparison, pros/cons, causes, or details, prefer bullets over one long paragraph.
- Keep very short factual answers to one concise paragraph.
"""

SUBQUESTION_PROMPT = """Given a user question and a list of document tools, output the minimum useful sub-questions in json markdown.

Rules:
- Generate at most ONE sub-question per tool/document.
- For broad questions about people, companies, impact, summaries, or comparisons, ask each document for the facts needed from that document.
- Do not split one document into many dimensions unless absolutely required.
- Use only tool names that appear in the tool list.
- Return ONLY a JSON markdown block.
- The JSON MUST be an object with a single "items" key.
- "items" MUST be a list of objects.
- Each object MUST have exactly these string keys: "sub_question" and "tool_name".
- Do NOT return a mapping like {"tool_name": "question"}.
- Do NOT add prose before or after the JSON.

<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
```json
{
  "items": [
    {
      "sub_question": "What does this document say that is relevant to the user's question?",
      "tool_name": "tool_name_here"
    }
  ]
}
```
"""


def _make_groq_llm():
    return OpenAILike(
        api_key=Config.GROQ_API_KEY,
        api_base="https://api.groq.com/openai/v1",
        model=Config.GROQ_SUBQUESTION_LLM,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=Config.GROQ_CONTEXT_WINDOW,
        temperature=0,
        max_tokens=Config.SUBQUESTION_MAX_TOKENS,
        max_retries=0,
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


def _run_async_query(engine, query: str):
    """Run a native LlamaIndex async query on the persistent provider loop."""
    return async_runtime.run(engine.aquery(query))


def _is_empty_response(response) -> bool:
    """LlamaIndex can swallow failed async sub-queries into Empty Response."""
    return str(response or "").strip().lower() in {"", "empty response", "none", "null"}


def _build_or_load_index(fname: str, upload_dir: Path, llm, embed_model=None):
    return shared_vector_index.build_or_load_file_index(fname, upload_dir, embed_model)


class _FallbackQueryEngine:
    """Use Gemini for just the document whose Groq answer is rate-limited."""

    def __init__(self, primary_engine, fallback_engine, filename: str):
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.filename = filename

    def __getattr__(self, name):
        """Preserve the LlamaIndex query-engine interface (e.g. callbacks)."""
        return getattr(self.primary_engine, name)

    def query(self, query):
        try:
            return self.primary_engine.query(query)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            logger.info("[subquestion] %s Groq rate-limited; answering this sub-question with Gemini", self.filename)
            return self.fallback_engine.query(query)

    async def aquery(self, query):
        try:
            return await self.primary_engine.aquery(query)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            logger.info("[subquestion] %s Groq rate-limited; answering this sub-question with Gemini", self.filename)
            return await self.fallback_engine.aquery(query)


def _make_tools(filenames: list[str], upload_dir: Path, llm, embed_model, fallback_llm=None):
    tools = []
    for fname in filenames:
        idx = _build_or_load_index(fname, upload_dir, llm, embed_model)
        primary_qe = idx.as_query_engine(similarity_top_k=3, llm=llm)
        qe = primary_qe
        if fallback_llm is not None:
            fallback_qe = idx.as_query_engine(similarity_top_k=3, llm=fallback_llm)
            qe = _FallbackQueryEngine(primary_qe, fallback_qe, fname)
        tools.append(
            QueryEngineTool(
                query_engine=qe,
                metadata=ToolMetadata(
                    name=Path(fname).stem.replace(" ", "_"),
                    description=f"Provides information about the document: {fname}",
                ),
            )
        )
    return tools


def _make_engine(tools, llm):
    Settings.llm = llm
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=LLMQuestionGenerator.from_defaults(
            llm=llm,
            prompt_template_str=SUBQUESTION_PROMPT,
        ),
        use_async=True,
        verbose=False,
    )


def _formatted_query(query: str) -> str:
    from answer_format import MATH_FORMAT_INSTRUCTIONS
    return f"{query.strip()}\n{ANSWER_FORMAT_INSTRUCTIONS}\n{MATH_FORMAT_INSTRUCTIONS}"


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    start = time.perf_counter()
    llm = _make_groq_llm()
    embed_model = model_cache.get_embed_model(Config.EMBED_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP
    logger.info(
        "[subquestion] LLM=OpenAILike/Groq (%s) use_async=True files=%s",
        Config.GROQ_SUBQUESTION_LLM,
        filenames,
    )

    fallback_llm = _make_gemini_llm()
    tools = _make_tools(filenames, upload_dir, llm, embed_model, fallback_llm)

    logger.info("[subquestion] index/tool setup took %.2fs", time.perf_counter() - start)

    query_start = time.perf_counter()
    try:
        response = _run_async_query(_make_engine(tools, llm), _formatted_query(query))
    except Exception as exc:
        from utils import is_rate_limit_error
        if not is_rate_limit_error(exc):
            raise
        logger.info(
            "[subquestion] Groq rate limit / quota exhausted (%s: %s); falling back to Gemini %s",
            type(exc).__name__,
            str(exc)[:240],
            Config.GOOGLE_LLM,
        )
        fallback_llm = _make_gemini_llm()
        fallback_tools = _make_tools(filenames, upload_dir, fallback_llm, embed_model)
        response = _run_async_query(
            _make_engine(fallback_tools, fallback_llm),
            _formatted_query(query),
        )

    # SubQuestionQueryEngine catches individual tool failures internally. If
    # every sub-question was rate-limited, no exception reaches the try/except
    # above and it returns "Empty Response". Retry the whole request with the
    # Gemini fallback rather than sending that misleading 200 response to UI.
    if _is_empty_response(response):
        logger.warning("[subquestion] all Groq sub-questions failed; retrying with Gemini %s", Config.GOOGLE_LLM)
        fallback_llm = _make_gemini_llm()
        fallback_tools = _make_tools(filenames, upload_dir, fallback_llm, embed_model)
        response = _run_async_query(
            _make_engine(fallback_tools, fallback_llm),
            _formatted_query(query),
        )
    logger.info("[subquestion] query execution took %.2fs", time.perf_counter() - query_start)

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [],
    }
