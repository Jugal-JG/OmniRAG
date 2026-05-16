import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class Config:
    APP_VERSION = os.getenv("APP_VERSION", "1.5")

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")          # Gemini models
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

    EMBED_MODEL = "BAAI/bge-base-en-v1.5"
    MISTRAL_EMBED_MODEL = "mistral-embed"

    # LLM model names. Env overrides let demos switch models without code edits.
    MISTRAL_LLM = os.getenv("MISTRAL_LLM", "mistral-large-latest")
    GOOGLE_LLM = os.getenv("GOOGLE_LLM", "gemini-3.1-flash-lite-preview")
    GOOGLE_GEMMA_LLM = os.getenv("GOOGLE_GEMMA_LLM", "gemma-4-26b-a4b-it")
    GEMINI_LLM = os.getenv("GEMINI_LLM", "gemini-2.5-flash")
    GROQ_LLM = os.getenv("GROQ_LLM", "meta-llama/llama-4-scout-17b-16e-instruct")
    GROQ_SUBQUESTION_LLM = os.getenv("GROQ_SUBQUESTION_LLM", "llama-3.3-70b-versatile")
    GOOGLE_MAX_RETRIES = _int_env("GOOGLE_MAX_RETRIES", 2)
    PDF_OCR_DPI = _int_env("PDF_OCR_DPI", 200)

    # Larger chunks = more context per retrieved node = better for structured PDFs
    # (tenant name tables, lease summary blocks all fit in one 1024-token chunk)
    CHUNK_SIZE = 1024
    # Retrieve more nodes so the first-page summary table is almost always included
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
