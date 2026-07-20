import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


class Config:
    APP_VERSION = os.getenv("APP_VERSION", "2")

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")          # Gemini models (primary)
    GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2", "")        # Multimodal key; ReAct fallback key (second account)
    GOOGLE_API_KEY_GEMMA = os.getenv("GOOGLE_API_KEY_GEMMA", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

    UPLOAD_FOLDER = os.getenv(
        "UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "uploads")
    )
    CACHE_FOLDER = os.getenv("CACHE_FOLDER", os.path.join(os.path.dirname(__file__), "cache"))

    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "None")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

    # [ORIGINAL] harrier-oss-v1-270m is a 270M-param *decoder* embedder — accurate but
    # heavy on CPU (seconds per 1024-token chunk). For faster CPU indexing set
    # EMBED_MODEL to a small sentence-transformer, e.g. BAAI/bge-small-en-v1.5
    # (33M) or sentence-transformers/all-MiniLM-L6-v2 (22M). The cache key
    # includes the model name, so switching auto-rebuilds without stale vectors.
    # EMBED_MODEL = os.getenv("EMBED_MODEL", "microsoft/harrier-oss-v1-270m")

    # [TEST] BAAI/bge-m3 — 567M-param encoder with native dense + sparse
    # retrieval. Supports up to 8192 tokens. Trained on a large multilingual
    # corpus including scientific text. Faster than harrier on CPU.
    MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    # Use Mistral's remote embedding API so HF CPU is not the indexing bottleneck.
    # The model name is included in the cache key, automatically isolating old BGE indexes.
    EMBED_MODEL = MISTRAL_EMBED_MODEL

    # LLM model names. Env overrides let demos switch models without code edits.
    MISTRAL_LLM = os.getenv("MISTRAL_LLM", "mistral-large-latest")
    ANSWER_MAX_TOKENS = _int_env("ANSWER_MAX_TOKENS", 4096)
    # Keep the same completion ceiling as other answer engines. This is a
    # maximum, not a requested response length; per-document Groq rate limits
    # are handled by the Gemini fallback in the Sub-Question engine.
    SUBQUESTION_MAX_TOKENS = _int_env("SUBQUESTION_MAX_TOKENS", ANSWER_MAX_TOKENS)
    # Merged answers combine two model results, so 1024 tokens can end mid-answer.
    MERGED_ANSWER_MAX_TOKENS = _int_env("MERGED_ANSWER_MAX_TOKENS", ANSWER_MAX_TOKENS)
    GOOGLE_LLM = os.getenv("GOOGLE_LLM", "gemini-3.1-flash-lite-preview")
    GOOGLE_GEMMA_LLM = os.getenv("GOOGLE_GEMMA_LLM", "gemma-4-26b-a4b-it")
    GEMINI_LLM = os.getenv("GEMINI_LLM", "gemini-2.5-flash")
    # ReAct primary model — uses gemini-2.5-flash on GOOGLE_API_KEY.
    # If that fails (429/quota), falls back to gemini-3.1-flash-lite-preview
    # on GOOGLE_API_KEY2 (second Google account).
    REACT_PRIMARY_LLM = os.getenv("REACT_PRIMARY_LLM", "gemini-2.5-flash")
    GROQ_LLM = os.getenv("GROQ_LLM", "qwen/qwen3.6-27b")
    # Kept for backward-compatible environment files; image analysis uses
    # GOOGLE_LLM with GOOGLE_API_KEY2 in engines/multimodal.py.
    GROQ_VISION_LLM = os.getenv("GROQ_VISION_LLM", "qwen/qwen3.6-27b")
    GROQ_ROUTER_LLM = os.getenv("GROQ_ROUTER_LLM", "qwen/qwen3.6-27b")
    GROQ_SUBQUESTION_LLM = os.getenv("GROQ_SUBQUESTION_LLM", "qwen/qwen3.6-27b")
    GROQ_CONTEXT_WINDOW = _int_env("GROQ_CONTEXT_WINDOW", 131072)
    GOOGLE_MAX_RETRIES = _int_env("GOOGLE_MAX_RETRIES", 2)
    PDF_OCR_DPI = _int_env("PDF_OCR_DPI", 200)

    # BAAI/bge-m3 supports up to 8,192 tokens, so the configured chunk is
    # embedded without truncation.
    # Shorter sequences reduce BGE-M3's CPU transformer cost substantially.
    CHUNK_SIZE = _int_env("CHUNK_SIZE", 512)
    # Overlap ensures formulas and sentences that span chunk boundaries
    # are captured in both neighbouring chunks.
    CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 64)
    # With Mistral's 1-RPS limit, larger sequential batches reduce request count.
    # 32 x 512-token chunks stays near the API's documented per-request token cap.
    EMBED_BATCH_SIZE = _int_env("EMBED_BATCH_SIZE", 32)
    PRELOAD_EMBED_MODEL = _bool_env("PRELOAD_EMBED_MODEL", True)
    # Retrieve the top-k most relevant chunks for the LLM to read.
    SIMILARITY_TOP_K = 8

    @classmethod
    def missing_keys(cls):
        missing = []
        if not cls.MISTRAL_API_KEY:
            missing.append("MISTRAL_API_KEY")
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        return missing
