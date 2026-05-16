"""
Sub-Question Query Engine — mirrors SubQuestion_Query_Engine.ipynb.
LLM: Groq Llama 3.3  |  Embeddings: HuggingFace BAAI/bge-base-en-v1.5
Decomposes complex queries into per-document sub-questions then synthesizes.
"""

from pathlib import Path
import logging
import time

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai_like import OpenAILike

import asyncio
import concurrent.futures

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, is_daily_quota_error, with_retry

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
        temperature=0,
        max_tokens=1024,
        max_retries=0,
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


def _run_async_in_thread(fn, *args, **kwargs):
    """Run LlamaIndex async internals in a clean event loop on Python 3.14."""
    async def runner():
        return await asyncio.to_thread(fn, *args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, runner()).result()


def _build_or_load_index(fname: str, upload_dir: Path, llm, embed_model=None):
    file_paths = [str(upload_dir / fname)]
    cache_path = index_cache.get_cache_path(file_paths, f"subquestion_{fname}")
    is_pdf = fname.lower().endswith(".pdf")

    if index_cache.is_cache_usable(file_paths, f"subquestion_{fname}", require_meta=is_pdf):
        logger.info("[subquestion] Loading cached vector index for %s", fname)
        ctx = StorageContext.from_defaults(persist_dir=str(cache_path))
        return load_index_from_storage(ctx)

    logger.info("[subquestion] Building vector index for %s", fname)
    from doc_loader import load_documents
    docs = load_documents(upload_dir / fname)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(cache_path))
    index_cache.save_documents_meta(file_paths, f"subquestion_{fname}", docs)
    return index


def _make_tools(filenames: list[str], upload_dir: Path, llm, embed_model):
    tools = []
    for fname in filenames:
        idx = _build_or_load_index(fname, upload_dir, llm, embed_model)
        qe = idx.as_query_engine(similarity_top_k=3, llm=llm)
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
    return f"{query.strip()}\n{ANSWER_FORMAT_INSTRUCTIONS}"


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    start = time.perf_counter()
    llm = _make_groq_llm()
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    logger.info(
        "[subquestion] LLM=OpenAILike/Groq (%s) use_async=True files=%s",
        Config.GROQ_SUBQUESTION_LLM,
        filenames,
    )

    tools = _make_tools(filenames, upload_dir, llm, embed_model)

    logger.info("[subquestion] index/tool setup took %.2fs", time.perf_counter() - start)

    query_start = time.perf_counter()
    try:
        response = _run_async_in_thread(_make_engine(tools, llm).query, _formatted_query(query))
    except Exception as exc:
        if not is_daily_quota_error(exc):
            raise
        logger.info(
            "[subquestion] Groq daily quota exhausted (%s: %s); falling back to Gemini %s",
            type(exc).__name__,
            str(exc)[:240],
            Config.GOOGLE_LLM,
        )
        fallback_llm = _make_gemini_llm()
        fallback_tools = _make_tools(filenames, upload_dir, fallback_llm, embed_model)
        response = _run_async_in_thread(
            _make_engine(fallback_tools, fallback_llm).query,
            _formatted_query(query),
        )
    logger.info("[subquestion] query execution took %.2fs", time.perf_counter() - query_start)

    sub_qa = []
    for sq in getattr(response, "metadata", {}).get("sub_question_response_pairs", []):
        sub_qa.append(f"Q: {sq.sub_q.sub_question}\nA: {sq.response}")

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": sub_qa,
    }
