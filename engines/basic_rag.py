"""Basic RAG engine using the shared BGE-M3 vector index."""

from pathlib import Path
import logging

from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI

import model_cache
import shared_vector_index
from config import Config
from utils import format_source_nodes, with_retry

logger = logging.getLogger(__name__)


def _build_or_load_index(file_paths: list[str], upload_dir: Path):
    embed_model = model_cache.get_hf_embed(Config.EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP
    return shared_vector_index.build_or_load_indexes(file_paths, upload_dir, embed_model)


@with_retry
def run(query: str, filenames: list[str], upload_dir: Path) -> dict:
    llm = MistralAI(api_key=Config.MISTRAL_API_KEY, model=Config.MISTRAL_LLM)
    Settings.llm = llm
    logger.info("[basic_rag] LLM=MistralAI (%s)", Config.MISTRAL_LLM)

    import retrieval

    index = _build_or_load_index(filenames, upload_dir)
    engine = retrieval.make_query_engine(
        index,
        similarity_top_k=Config.SIMILARITY_TOP_K,
        llm=llm,
    )
    response = engine.query(query)
    return {
        "answer": str(response),
        "sources": format_source_nodes(getattr(response, "source_nodes", [])),
        "thinking_steps": [],
    }
