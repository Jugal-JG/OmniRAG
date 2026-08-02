"""Small persistent registry for the owner-only OmniRAG admin dashboard."""

import sqlite3
import time
from pathlib import Path

from config import Config


def _database_path() -> Path:
    path = Path(Config.CACHE_FOLDER) / "admin" / "users.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS users (
            account_key TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT NOT NULL,
            first_seen INTEGER NOT NULL,
            last_seen INTEGER NOT NULL
        )"""
    )
    return connection


def record_user(account_key: str, claims: dict) -> None:
    """Upsert a profile after Google has verified its identity token."""
    now = int(time.time())
    email = str(claims.get("email") or "").strip().lower()
    name = str(claims.get("name") or email or "OmniRAG user").strip()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO users (account_key, email, name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(account_key) DO UPDATE SET
                   email=excluded.email,
                   name=excluded.name,
                   last_seen=excluded.last_seen""",
            (account_key, email, name, now, now),
        )


def users() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT account_key, email, name, first_seen, last_seen FROM users ORDER BY last_seen DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def forget_user(account_key: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM users WHERE account_key = ?", (account_key,))
