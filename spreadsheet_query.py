"""Safe natural-language query planning over the local spreadsheet row store."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
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


def _normalised_name(value: str) -> str:
    """Normalize a header or phrase for schema-driven matching."""
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _best_mentioned_column(query: str, columns: list[str]) -> str | None:
    """Match the requested field to a real header, allowing small typos.

    This is deliberately based on the workbook schema rather than a list of
    known financial columns.  A request such as "highest yearly income",
    "lowest credit score", or a misspelled custom header is resolved from the
    headers in the uploaded file.
    """
    words = re.findall(r"[a-z0-9]+", query.lower())
    matches: list[tuple[float, int, str]] = []
    for column in columns:
        header_words = re.findall(r"[a-z0-9]+", column.lower())
        if not header_words:
            continue
        width = len(header_words)
        header = _normalised_name(column)
        for start in range(max(1, len(words) - width + 1)):
            phrase = _normalised_name(" ".join(words[start:start + width]))
            score = SequenceMatcher(None, header, phrase).ratio()
            if score >= 0.82:
                matches.append((score, len(header), column))
    if not matches:
        return None
    # Longer headers win ties so a specific field beats a generic "income".
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def _deterministic_extreme(query: str, schemas: list[dict], rows: list[dict]) -> dict | None:
    """Answer obvious highest/lowest-column requests without an LLM planner."""
    lowered = query.lower()
    if not re.search(r"\b(?:highest|largest|maximum|max|lowest|smallest|minimum|min)\b", lowered):
        return None
    operation = "min" if re.search(r"\b(?:lowest|smallest|minimum|min)\b", lowered) else "max"
    columns = list(dict.fromkeys(column for schema in schemas for column in schema["columns"]))
    column = _best_mentioned_column(lowered, columns)
    if column is None:
        return None
    numeric_rows = [(row, _number(row["values"].get(column))) for row in rows]
    numeric_rows = [(row, value) for row, value in numeric_rows if value is not None]
    if not numeric_rows:
        return None
    selected_row, value = (min(numeric_rows, key=lambda item: item[1]) if operation == "min" else max(numeric_rows, key=lambda item: item[1]))
    identifier = next((key for key in ("id", "user_id", "user id") if key in selected_row["values"]), None)
    subject = f"User ID **{selected_row['values'][identifier]}** " if identifier else "The matching row "
    return {
        "answer": (
            f"{subject}has the exact {operation}imum **{column}**: "
            f"**{value:,}** ({selected_row['file']}, {selected_row['sheet']} row {selected_row['row']})."
        ),
        "sources": [{"file": selected_row["file"], "text": f"{selected_row['sheet']} row {selected_row['row']}", "score": 1.0}],
        "thinking_steps": ["Answered by scanning every numeric value in the local structured spreadsheet store."],
    }


def try_structured_query(query: str, filenames: list[str], upload_dir: Path) -> dict | None:
    """Return an exact SQLite-backed result, or None for semantic questions."""
    schemas, rows = structured_snapshot(filenames, upload_dir)
    if not schemas:
        return None
    deterministic_result = _deterministic_extreme(query, schemas, rows)
    if deterministic_result is not None:
        return deterministic_result
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
