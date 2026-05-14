"""
Sub-Question Query Engine — mirrors SubQuestion_Query_Engine.ipynb.
LLM: Google Gemma 4-31b  |  Embeddings: HuggingFace BAAI/bge-base-en-v1.5
Decomposes complex queries into per-document sub-questions then synthesizes.
"""

from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.google_genai import GoogleGenAI

import index_cache
import model_cache
from config import Config
from utils import format_source_nodes, with_retry


def _build_or_load_index(fname: str, upload_dir: Path, llm, embed_model=None):
    file_paths = [fname]
    cache_path = index_cache.get_cache_path(file_paths, f"subquestion_{fname}")
    is_pdf = fname.lower().endswith(".pdf")

    if not is_pdf and index_cache.is_cached(file_paths, f"subquestion_{fname}"):
        ctx = StorageContext.from_defaults(persist_dir=str(cache_path))
        return load_index_from_storage(ctx)

    from doc_loader import load_documents
    docs = load_documents(upload_dir / fname)
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(cache_path))
    return index


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    llm = GoogleGenAI(api_key=Config.GOOGLE_API_KEY_GEMMA, model=Config.GOOGLE_LLM, max_retries=5)
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE

    tools = []
    for fname in filenames:
        idx = _build_or_load_index(fname, upload_dir, llm, embed_model)
        qe = idx.as_query_engine(similarity_top_k=5)
        tools.append(
            QueryEngineTool(
                query_engine=qe,
                metadata=ToolMetadata(
                    name=Path(fname).stem.replace(" ", "_"),
                    description=f"Provides information about the document: {fname}",
                ),
            )
        )

    # use_async=False + sync query avoids all event-loop conflicts with Flask
    engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=LLMQuestionGenerator.from_defaults(llm=llm),
        use_async=False,
        verbose=True,
    )

    response = engine.query(query)

    sub_qa = []
    for sq in getattr(response, "metadata", {}).get("sub_question_response_pairs", []):
        sub_qa.append(f"Q: {sq.sub_q.sub_question}\nA: {sq.response}")

    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": sub_qa,
    }
