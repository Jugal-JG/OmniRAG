"""Safe natural-language query planning over the local spreadsheet row store."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from llama_index.llms.google_genai import GoogleGenAI

from config import Config
from spreadsheet_store import structured_snapshot

_OPERATIONS = {"rows", "count", "sum", "average", "min", "max"}
_FILTER_OPS = {"eq", "ne", "contains", "gt", "gte", "lt", "lte"}


def _llm() -> GoogleGenAI:
    return GoogleGenAI(
        api_key=Config.GOOGLE_API_KEY,
        model=Config.GOOGLE_LLM,
        temperature=0,
        max_tokens=700,
        max_retries=Config.GOOGLE_MAX_RETRIES,
        is_function_calling_model=False,
    )


def _json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)


def _number(value) -> Decimal | None:
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    cleaned = re.sub(r"[^0-9.()\-]", "", str(value).replace(",", ""))
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _resolve_column(requested: str | None, columns: list[str]) -> str | None:
    if not requested:
        return None
    lowered = requested.strip().lower()
    exact = next((column for column in columns if column.lower() == lowered), None)
    if exact:
        return exact
    return next((column for column in columns if lowered in column.lower()), None)


def _matches(value, operator: str, expected) -> bool:
    if operator in {"gt", "gte", "lt", "lte"}:
        left, right = _number(value), _number(expected)
        if left is None or right is None:
            return False
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[operator]
    left, right = str(value).strip().lower(), str(expected).strip().lower()
    if operator == "contains":
        return right in left
    if operator == "ne":
        return left != right
    return left == right


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    shown = columns[:8]
    header = "| " + " | ".join(shown) + " |"
    divider = "| " + " | ".join("---" for _ in shown) + " |"
    body = []
    for row in rows[:20]:
        values = [str(row["values"].get(column, "")).replace("|", "\\|") for column in shown]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def try_structured_query(query: str, filenames: list[str], upload_dir: Path) -> dict | None:
    """Return an exact SQLite-backed result, or None for semantic questions."""
    schemas, rows = structured_snapshot(filenames, upload_dir)
    if not schemas:
        return None
    schema_text = json.dumps(schemas, ensure_ascii=False)
    prompt = f"""Classify and plan this spreadsheet question using only the supplied schema.
Return JSON only. Use mode "structured" for exact row lookup, filtering, counting,
sum, average, minimum, or maximum. Use mode "semantic" for summaries, explanations,
themes, opinions, forecasting, or questions requiring narrative interpretation.

For structured mode return:
{{"mode":"structured","operation":"rows|count|sum|average|min|max","value_column":null,
 "filters":[{{"column":"exact schema column","operator":"eq|ne|contains|gt|gte|lt|lte","value":"..."}}],
 "sheet":null,"limit":20}}
Never invent a column. For count, value_column must be null.

Schema: {schema_text}
Question: {query}
"""
    try:
        plan = _json_object(str(_llm().complete(prompt)))
    except Exception:
        return None
    if plan.get("mode") != "structured" or plan.get("operation") not in _OPERATIONS:
        return None

    all_columns = list(dict.fromkeys(column for schema in schemas for column in schema["columns"]))
    sheet = plan.get("sheet")
    candidates = [row for row in rows if not sheet or row["sheet"].lower() == str(sheet).lower()]
    for condition in plan.get("filters", [])[:8]:
        operator = condition.get("operator", "eq")
        column = _resolve_column(condition.get("column"), all_columns)
        if operator not in _FILTER_OPS or not column:
            return None
        candidates = [
            row for row in candidates
            if column in row["values"] and _matches(row["values"][column], operator, condition.get("value", ""))
        ]

    operation = plan["operation"]
    sources = [{"file": row["file"], "text": f"{row['sheet']} row {row['row']}", "score": 1.0} for row in candidates[:8]]
    if operation == "rows":
        limit = max(1, min(int(plan.get("limit") or 20), 50))
        answer = f"Found **{len(candidates)} matching row(s)**.\n\n" + _markdown_table(candidates[:limit], all_columns)
    elif operation == "count":
        answer = f"The exact matching row count is **{len(candidates):,}**."
    else:
        column = _resolve_column(plan.get("value_column"), all_columns)
        if not column:
            return None
        values = [(row, _number(row["values"].get(column))) for row in candidates]
        values = [(row, value) for row, value in values if value is not None]
        if not values:
            answer = f"No numeric values were found in **{column}** for the matching rows."
        elif operation == "sum":
            answer = f"The exact sum of **{column}** is **{sum(value for _, value in values):,}** across {len(values):,} row(s)."
        elif operation == "average":
            answer = f"The exact average of **{column}** is **{sum(value for _, value in values) / len(values):,.4f}** across {len(values):,} row(s)."
        else:
            selected_row, selected_value = (min(values, key=lambda item: item[1]) if operation == "min" else max(values, key=lambda item: item[1]))
            answer = f"The exact {operation} of **{column}** is **{selected_value:,}** ({selected_row['sheet']}, row {selected_row['row']})."
    return {"answer": answer, "sources": sources, "thinking_steps": ["Answered from the local structured spreadsheet store; vector embeddings were not required."]}
