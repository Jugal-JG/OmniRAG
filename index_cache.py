"""Disk-based index caching keyed by SHA-256 of file contents + engine name."""

import hashlib
import json
import os
from pathlib import Path

from config import Config


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
    # Include chunk_size in the key so changing it auto-invalidates old indexes.
    chunk_tag = f"c{Config.CHUNK_SIZE}"
    return f"{engine_name}_{chunk_tag}_{_file_hash(file_paths)}"


def cache_dir(key: str) -> Path:
    return Path(Config.CACHE_FOLDER) / key


def is_cached(file_paths: list[str], engine_name: str) -> bool:
    d = cache_dir(cache_key(file_paths, engine_name))
    return d.exists() and (d / "docstore.json").exists()


def get_cache_path(file_paths: list[str], engine_name: str) -> Path:
    key = cache_key(file_paths, engine_name)
    d = cache_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_index_meta(file_paths: list[str], engine_name: str, meta: dict):
    """Save arbitrary JSON metadata alongside the index."""
    d = get_cache_path(file_paths, engine_name)
    (d / "meta.json").write_text(json.dumps(meta))


def load_index_meta(file_paths: list[str], engine_name: str) -> dict:
    d = cache_dir(cache_key(file_paths, engine_name))
    meta_file = d / "meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return {}
