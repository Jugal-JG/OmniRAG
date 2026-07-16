import asyncio
import io
import contextlib
import time
import functools
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
TEXT_EXTENSIONS = {".txt", ".pdf", ".html", ".md"}


def is_rate_limit_error(exc):
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg


def is_daily_quota_error(exc):
    msg = str(exc).lower()
    daily_markers = (
        "tokens per day",
        "requests per day",
        " tpd",
        " rpd",
        "daily token",
        "daily request",
    )
    return any(marker in msg for marker in daily_markers)


def with_retry(fn):
    """Decorator: retry up to 3 times on rate-limit errors with exponential backoff."""

    @functools.wraps(fn)
    @retry(
        retry=retry_if_exception(
            lambda exc: is_rate_limit_error(exc) and not is_daily_quota_error(exc)
        ),
        wait=wait_exponential(multiplier=1, min=3, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def capture_stdout(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), capturing anything printed to stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def classify_files(filenames):
    """Split a list of filenames into (images, texts)."""
    from pathlib import Path

    images = [f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
    texts = [f for f in filenames if Path(f).suffix.lower() in TEXT_EXTENSIONS]
    return images, texts


def format_source_nodes(source_nodes):
    """Extract readable source info from LlamaIndex NodeWithScore list."""
    sources = []
    for node in source_nodes or []:
        text = ""
        file_name = ""
        score = None
        if hasattr(node, "node"):
            text = node.node.get_content()[:300]
            file_name = node.node.metadata.get("file_name", "")
            score = getattr(node, "score", None)
            if score is None:
                metadata = getattr(node.node, "metadata", None) or {}
                score = metadata.get("score", metadata.get("similarity", metadata.get("relevance_score")))
        elif hasattr(node, "get_content"):
            text = node.get_content()[:300]
            metadata = getattr(node, "metadata", None) or {}
            file_name = metadata.get("file_name", "")
            score = metadata.get("score", metadata.get("similarity", metadata.get("relevance_score")))
        try:
            score = round(float(score), 4) if score is not None else None
        except (TypeError, ValueError):
            score = None
        sources.append({"text": text, "file": file_name, "score": score})
    return sources


def rate_sleep(seconds=1.0):
    """Pause briefly to avoid back-to-back requests to the same provider."""
    time.sleep(seconds)


def ensure_event_loop():
    """
    Guarantee a live asyncio event loop exists in the current thread.

    LlamaIndex agents and query engines call asyncio.get_event_loop() internally
    even from their *synchronous* APIs (chat(), query(), etc.).  In Python 3.10+
    that raises RuntimeError('no running event loop') if no loop has been set for
    the thread.  Calling this at the top of every engine run() fixes it cheaply.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop is closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
