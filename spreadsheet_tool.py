"""Exact spreadsheet tool shared by agent-based query engines."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.tools import QueryEngineTool


SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}


def has_spreadsheets(filenames: list[str]) -> bool:
    return any(Path(name).suffix.lower() in SPREADSHEET_EXTENSIONS for name in filenames)


class StructuredSpreadsheetQueryEngine(CustomQueryEngine):
    """Adapter that lets query-engine and agent workflows use the row store."""

    filenames: list[str]
    upload_dir: str

    def custom_query(self, query_str: str) -> str:
        from spreadsheet_query import try_structured_query

        result = try_structured_query(query_str, self.filenames, Path(self.upload_dir))
        if result is None:
            return (
                "No exact spreadsheet operation was recognized. Use document search "
                "only for narrative spreadsheet questions, not for maxima, minima, "
                "counts, sums, averages, filters, or exact row lookups."
            )
        return result["answer"]


def make_spreadsheet_query_tool(filenames: list[str], upload_dir: Path, name: str = "query_spreadsheet_data") -> QueryEngineTool | None:
    """Create a query-engine tool for exact filters and full-sheet aggregations."""
    spreadsheet_names = [
        filename for filename in filenames
        if Path(filename).suffix.lower() in SPREADSHEET_EXTENSIONS
    ]
    if not spreadsheet_names:
        return None

    return QueryEngineTool.from_defaults(
        query_engine=StructuredSpreadsheetQueryEngine(
            filenames=spreadsheet_names,
            upload_dir=str(upload_dir),
        ),
        name=name,
        description=(
            "Queries the complete uploaded CSV/XLSX data exactly. Use this for any "
            "spreadsheet maximum, minimum, count, sum, average, filter, or row lookup. "
            "It is the authoritative source for numerical spreadsheet answers; do not "
            "infer an aggregate from document-search excerpts."
        ),
    )
