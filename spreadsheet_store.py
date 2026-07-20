"""Persistent structured and semantic storage for CSV/XLSX workbooks."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from config import Config

SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}


def file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _database_path() -> Path:
    path = Path(Config.CACHE_FOLDER) / "spreadsheets.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS spreadsheet_files (
            file_hash TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS spreadsheet_sheets (
            file_hash TEXT NOT NULL,
            file_name TEXT NOT NULL,
            sheet_name TEXT NOT NULL,
            headers_json TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            formula_count INTEGER NOT NULL,
            sample_json TEXT NOT NULL,
            PRIMARY KEY (file_hash, sheet_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS spreadsheet_rows (
            file_hash TEXT NOT NULL,
            file_name TEXT NOT NULL,
            sheet_name TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            values_json TEXT NOT NULL,
            formulas_json TEXT NOT NULL,
            row_text TEXT NOT NULL,
            PRIMARY KEY (file_hash, sheet_name, row_number)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_sheet_rows_file ON spreadsheet_rows(file_hash, sheet_name)"
    )
    return connection


def begin_ingestion(file_path: Path) -> None:
    digest = file_hash(file_path)
    with _connect() as connection:
        connection.execute(
            """INSERT INTO spreadsheet_files (file_hash, file_name, status)
               VALUES (?, ?, 'parsing')
               ON CONFLICT(file_hash) DO UPDATE SET
                 file_name=excluded.file_name, status='parsing', error='',
                 updated_at=CURRENT_TIMESTAMP""",
            (digest, file_path.name),
        )


def finish_ingestion(file_path: Path) -> int:
    digest = file_hash(file_path)
    with _connect() as connection:
        count = int(connection.execute(
            "SELECT COUNT(*) FROM spreadsheet_rows WHERE file_hash = ?", (digest,)
        ).fetchone()[0])
        connection.execute(
            "UPDATE spreadsheet_files SET status='ready', row_count=?, error='', updated_at=CURRENT_TIMESTAMP WHERE file_hash=?",
            (count, digest),
        )
    return count


def fail_ingestion(file_path: Path, error: Exception | str) -> None:
    digest = file_hash(file_path)
    with _connect() as connection:
        connection.execute(
            """INSERT INTO spreadsheet_files (file_hash, file_name, status, error)
               VALUES (?, ?, 'failed', ?)
               ON CONFLICT(file_hash) DO UPDATE SET status='failed', error=excluded.error,
                 updated_at=CURRENT_TIMESTAMP""",
            (digest, file_path.name, str(error)[:1000]),
        )


def is_ready(file_path: Path) -> bool:
    if not file_path.exists() or file_path.suffix.lower() not in SPREADSHEET_EXTENSIONS:
        return False
    digest = file_hash(file_path)
    with _connect() as connection:
        row = connection.execute(
            "SELECT status FROM spreadsheet_files WHERE file_hash = ?", (digest,)
        ).fetchone()
    return bool(row and row[0] == "ready")


def index_sheet(
    file_path: Path,
    sheet_name: str,
    headers: list[str],
    rows: Iterable[tuple],
    formula_rows: Iterable[tuple] | None = None,
) -> None:
    """Replace a sheet in SQLite while preserving its original row structure."""
    digest = file_hash(file_path)
    formula_iter = iter(formula_rows) if formula_rows is not None else None
    records = []
    samples = []
    formula_count = 0

    for row_number, row in enumerate(rows, start=2):
        formula_row = next(formula_iter, ()) if formula_iter is not None else ()
        values: dict[str, str] = {}
        formulas: dict[str, str] = {}
        text_parts = [f"Spreadsheet: {file_path.name}", f"Sheet: {sheet_name}", f"Row: {row_number}"]
        for column, value in enumerate(row):
            if value is None or str(value).strip() == "":
                continue
            header = headers[column] if column < len(headers) else f"Column {column + 1}"
            values[header] = value.isoformat() if hasattr(value, "isoformat") else str(value)
            text_parts.append(f"{header}={values[header]}")
            formula = formula_row[column] if column < len(formula_row) else None
            if isinstance(formula, str) and formula.startswith("="):
                formulas[header] = formula
                formula_count += 1
                text_parts.append(f"{header} formula={formula}")
        if values:
            if len(samples) < 3:
                samples.append(values)
            records.append((
                digest, file_path.name, sheet_name, row_number,
                json.dumps(values, ensure_ascii=False, default=str),
                json.dumps(formulas, ensure_ascii=False),
                " | ".join(text_parts),
            ))

    with _connect() as connection:
        connection.execute(
            "DELETE FROM spreadsheet_rows WHERE file_hash = ? AND sheet_name = ?",
            (digest, sheet_name),
        )
        connection.executemany(
            """INSERT INTO spreadsheet_rows
               (file_hash, file_name, sheet_name, row_number, values_json, formulas_json, row_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records,
        )
        connection.execute(
            """INSERT INTO spreadsheet_sheets
               (file_hash, file_name, sheet_name, headers_json, row_count, formula_count, sample_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(file_hash, sheet_name) DO UPDATE SET
                 headers_json=excluded.headers_json, row_count=excluded.row_count,
                 formula_count=excluded.formula_count, sample_json=excluded.sample_json""",
            (digest, file_path.name, sheet_name, json.dumps(headers, ensure_ascii=False),
             len(records), formula_count, json.dumps(samples, ensure_ascii=False)),
        )


def cached_row_count(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    digest = file_hash(file_path)
    with _connect() as connection:
        row = connection.execute(
            "SELECT row_count FROM spreadsheet_files WHERE file_hash = ? AND status='ready'",
            (digest,),
        ).fetchone()
    return int(row[0]) if row else 0


def semantic_records(file_path: Path) -> list[dict]:
    """Return a small semantic representation instead of embedding every numeric row."""
    digest = file_hash(file_path)
    with _connect() as connection:
        sheets = connection.execute(
            """SELECT sheet_name, headers_json, row_count, formula_count, sample_json
               FROM spreadsheet_sheets WHERE file_hash=? ORDER BY sheet_name""",
            (digest,),
        ).fetchall()
        formula_rows = connection.execute(
            """SELECT sheet_name, row_number, formulas_json FROM spreadsheet_rows
               WHERE file_hash=? AND formulas_json != '{}' ORDER BY sheet_name, row_number""",
            (digest,),
        ).fetchall()
        narrative_rows = connection.execute(
            """SELECT sheet_name, row_number, values_json FROM spreadsheet_rows
               WHERE file_hash=? ORDER BY sheet_name, row_number""",
            (digest,),
        ).fetchall()

    records: list[dict] = []
    for sheet_name, headers_json, row_count, formula_count, sample_json in sheets:
        headers = json.loads(headers_json)
        samples = json.loads(sample_json)
        records.append({
            "text": (
                f"Spreadsheet overview: {file_path.name}\nSheet: {sheet_name}\n"
                f"Rows: {row_count}\nColumns: {' | '.join(headers)}\n"
                f"Formula cells: {formula_count}\nSample rows: {json.dumps(samples, ensure_ascii=False)}"
            ),
            "metadata": {"file_name": file_path.name, "file_path": str(file_path), "sheet": sheet_name, "kind": "sheet_profile"},
        })

    for start in range(0, len(formula_rows), 50):
        batch = formula_rows[start:start + 50]
        text = "\n".join(
            f"Sheet {sheet}, row {row}: {formulas}" for sheet, row, formulas in batch
        )
        records.append({
            "text": f"Spreadsheet formulas: {file_path.name}\n{text}",
            "metadata": {"file_name": file_path.name, "file_path": str(file_path), "kind": "formulas"},
        })

    narrative = []
    for sheet, row, values_json in narrative_rows:
        values = json.loads(values_json)
        if any(len(str(value)) >= 80 for value in values.values()):
            narrative.append(f"Sheet {sheet}, row {row}: {json.dumps(values, ensure_ascii=False)}")
    for start in range(0, len(narrative), 30):
        records.append({
            "text": f"Narrative spreadsheet cells: {file_path.name}\n" + "\n".join(narrative[start:start + 30]),
            "metadata": {"file_name": file_path.name, "file_path": str(file_path), "kind": "narrative_rows"},
        })
    return records


def profile_context(filenames: list[str], upload_dir: Path) -> str:
    chunks = []
    for name in filenames:
        path = upload_dir / name
        if path.suffix.lower() not in SPREADSHEET_EXTENSIONS or not is_ready(path):
            return ""
        chunks.extend(record["text"] for record in semantic_records(path) if record["metadata"]["kind"] == "sheet_profile")
    return "\n\n".join(chunks)


def structured_snapshot(filenames: list[str], upload_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return validated rows for ready spreadsheets among the selected files.

    A document set may legitimately contain a PDF alongside a workbook.  The
    previous all-or-nothing check discarded the workbook in that case, forcing
    callers to answer numeric questions from vector-search samples instead of
    the complete local table.
    """
    paths = [
        upload_dir / name
        for name in filenames
        if (upload_dir / name).suffix.lower() in SPREADSHEET_EXTENSIONS
    ]
    if not paths or any(not is_ready(path) for path in paths):
        return [], []
    hashes = [file_hash(path) for path in paths]
    placeholders = ",".join("?" for _ in hashes)
    with _connect() as connection:
        raw_schemas = connection.execute(
            f"""SELECT file_name, sheet_name, headers_json, row_count
                FROM spreadsheet_sheets WHERE file_hash IN ({placeholders})
                ORDER BY file_name, sheet_name""",
            hashes,
        ).fetchall()
        raw_rows = connection.execute(
            f"""SELECT file_name, sheet_name, row_number, values_json
                FROM spreadsheet_rows WHERE file_hash IN ({placeholders})
                ORDER BY file_name, sheet_name, row_number""",
            hashes,
        ).fetchall()
    schemas = [
        {"file": file_name, "sheet": sheet, "columns": json.loads(headers), "rows": count}
        for file_name, sheet, headers, count in raw_schemas
    ]
    rows = [
        {"file": file_name, "sheet": sheet, "row": row_number, "values": json.loads(values)}
        for file_name, sheet, row_number, values in raw_rows
    ]
    return schemas, rows


def relevant_rows(filenames: list[str], upload_dir: Path, query: str, limit: int = 12) -> str:
    paths = [upload_dir / name for name in filenames if (upload_dir / name).suffix.lower() in SPREADSHEET_EXTENSIONS and (upload_dir / name).exists()]
    if not paths:
        return ""
    hashes = [file_hash(path) for path in paths]
    placeholders = ",".join("?" for _ in hashes)
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT file_name, sheet_name, row_number, row_text FROM spreadsheet_rows WHERE file_hash IN ({placeholders})",
            hashes,
        ).fetchall()
    terms = set(re.findall(r"[a-z0-9]{2,}", query.lower()))
    scored = []
    for row in rows:
        score = sum(term in row[3].lower() for term in terms)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2]))
    return "\n".join(
        f"[{file_name} | {sheet_name} | row {row_number}] {row_text}"
        for _, (file_name, sheet_name, row_number, row_text) in scored[:limit]
    )
