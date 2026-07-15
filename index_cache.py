"""Disk-based index caching keyed by SHA-256 of file contents + engine name."""

import hashlib
import json
import os
import threading
from pathlib import Path

from config import Config

# ── Build locks ───────────────────────────────────────────────────────────────
# One lock per cache key so that when two threads want the SAME index (e.g. the
# background pre-indexer started on upload AND the first user query), only one
# actually embeds the documents. The other blocks on the lock, then finds the
# index already persisted and loads it from disk instead of re-embedding.
_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def build_lock(file_paths: list[str], engine_name: str) -> threading.Lock:
    """Return the process-wide lock guarding builds for this cache key."""
    key = cache_key(file_paths, engine_name)
    with _build_locks_guard:
        lock = _build_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _build_locks[key] = lock
        return lock


def _file_hash(file_paths: list[str]) -> str:
    """Stable hash over sorted file contents."""
    h = hashlib.sha256()
    for fp in sorted(file_paths):
        h.update(Path(fp).name.encode())
        try:
            h.update(Path(fp).read_bytes())
        except Exception:
            pass
    return h.hexdigest()[:16]


def cache_key(file_paths: list[str], engine_name: str) -> str:
    # Include chunk_size and embed model in the key so changing either
    # auto-invalidates old indexes (avoids stale vector mismatches).
    chunk_tag = f"c{Config.CHUNK_SIZE}"
    model_tag = Config.EMBED_MODEL.replace("/", "_")
    return f"{engine_name}_{chunk_tag}_{model_tag}_{_file_hash(file_paths)}"


def cache_dir(key: str) -> Path:
    return Path(Config.CACHE_FOLDER) / key


def is_cached(file_paths: list[str], engine_name: str) -> bool:
    d = cache_dir(cache_key(file_paths, engine_name))
    return d.exists() and (d / "docstore.json").exists()


def document_cache_meta(docs) -> dict:
    """Build validation metadata for cached indexes."""
    text_chars = 0
    for doc in docs or []:
        try:
            text = doc.get_content()
        except Exception:
            text = getattr(doc, "text", "") or ""
        text_chars += len(text.strip())
    return {"doc_count": len(docs or []), "text_chars": text_chars}


def is_cache_usable(file_paths: list[str], engine_name: str, *, require_meta: bool = False) -> bool:
    """Return True when a persisted index exists and its metadata is safe to reuse."""
    if not is_cached(file_paths, engine_name):
        return False

    meta = load_index_meta(file_paths, engine_name)
    if not meta:
        return not require_meta

    return meta.get("doc_count", 0) > 0 and meta.get("text_chars", 0) > 0


def get_cache_path(file_paths: list[str], engine_name: str) -> Path:
    key = cache_key(file_paths, engine_name)
    d = cache_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_index_meta(file_paths: list[str], engine_name: str, meta: dict):
    """Save arbitrary JSON metadata alongside the index."""
    d = get_cache_path(file_paths, engine_name)
    (d / "meta.json").write_text(json.dumps(meta))


def save_documents_meta(file_paths: list[str], engine_name: str, docs):
    """Save document validation metadata beside a persisted index."""
    save_index_meta(file_paths, engine_name, document_cache_meta(docs))


def load_index_meta(file_paths: list[str], engine_name: str) -> dict:
    d = cache_dir(cache_key(file_paths, engine_name))
    meta_file = d / "meta.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            return {}
    return {}
