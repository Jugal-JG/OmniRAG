"""Safe natural-language query planning over the local spreadsheet row store."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from decimal import Decimal, InvalidOperation
from pathlib import Path

from config import Config
from spreadsheet_store import structured_snapshot

_OPERATIONS = {"rows", "count", "sum", "average", "min", "max", "group", "join_group"}
_FILTER_OPS = {"eq", "ne", "contains", "gt", "gte", "lt", "lte"}


def spreadsheet_schema_issue(filenames: list[str], upload_dir: Path) -> str | None:
    """Return a user-safe error when a report sheet has no trustworthy headers."""
    schemas, _ = structured_snapshot(filenames, upload_dir)
    for schema in schemas:
        placeholders = [column for column in schema["columns"] if re.fullmatch(r"Column \d+", column)]
        if placeholders:
            return (
                f"{schema['file']} has unnamed fields ({', '.join(placeholders[:3])}). "
                "It appears to be a report-style sheet whose real table header was not identified, "
                "so trend analysis has been stopped rather than using unreliable sample data."
            )
    return None


def _llm():
    from llama_index.llms.google_genai import GoogleGenAI

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


def _is_missing(value) -> bool:
    """Treat common report-export missing markers as missing, not categories."""
    return value is None or str(value).strip().lower() in {"", ".", "-", "n/a", "na", "null", "none"}


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


def _markdown_table(rows: list[dict], columns: list[str], row_limit: int = 20) -> str:
    # Do not silently discard fields.  The browser makes wide Markdown tables
    # horizontally scrollable, so a CSV's complete schema remains available.
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows[:row_limit]:
        values = [str(row["values"].get(column, "")).replace("|", "\\|") for column in columns]
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


def _grouped_result(plan: dict, schemas: list[dict], rows: list[dict]) -> dict | None:
    """Execute a schema-planned aggregation across every matching spreadsheet row."""
    all_columns = list(dict.fromkeys(column for schema in schemas for column in schema["columns"]))
    group_by = _resolve_column(plan.get("group_by"), all_columns)
    requested_metrics = plan.get("metrics") or [plan.get("value_column")]
    metrics = list(dict.fromkeys(_resolve_column(metric, all_columns) for metric in requested_metrics if metric))
    metrics = [metric for metric in metrics if metric and metric != group_by]
    aggregation = str(plan.get("aggregation") or "average").lower()
    if not group_by or not metrics or aggregation not in {"average", "sum", "min", "max", "count"}:
        return None

    bin_size = plan.get("bin_size")
    try:
        bin_size = int(bin_size) if bin_size is not None else None
    except (TypeError, ValueError):
        return None
    if bin_size is not None and not 2 <= bin_size <= 100:
        return None

    groups: dict[str, list[dict]] = {}
    for row in rows:
        raw_group = row["values"].get(group_by)
        numeric_group = _number(raw_group)
        if raw_group in (None, ""):
            continue
        if bin_size and numeric_group is not None:
            start = int(numeric_group // bin_size) * bin_size
            label = f"{start}\u2013{start + bin_size - 1}"
        else:
            label = str(raw_group)
        groups.setdefault(label, []).append(row)
    if not groups:
        return None

    def group_sort(item: tuple[str, list[dict]]):
        numeric = _number(item[0].split("\u2013", 1)[0])
        return (numeric is None, numeric if numeric is not None else item[0].lower())

    values_by_group: dict[str, dict[str, Decimal]] = {}
    table_rows = []
    for label, group_rows in sorted(groups.items(), key=group_sort):
        rendered = {group_by: label, "records": len(group_rows)}
        values_by_group[label] = {}
        for metric in metrics:
            numbers = [_number(row["values"].get(metric)) for row in group_rows]
            numbers = [number for number in numbers if number is not None]
            if aggregation == "count":
                result = Decimal(len(numbers))
            elif not numbers:
                continue
            elif aggregation == "sum":
                result = sum(numbers)
            elif aggregation == "min":
                result = min(numbers)
            elif aggregation == "max":
                result = max(numbers)
            else:
                result = sum(numbers) / len(numbers)
            values_by_group[label][metric] = result
            rendered[metric] = f"{result:,.2f}"
        table_rows.append({"values": rendered})
    if not any(values_by_group.values()):
        return None

    summary_parts = []
    for metric in metrics:
        candidates = [(label, results[metric]) for label, results in values_by_group.items() if metric in results]
        if candidates:
            label, value = max(candidates, key=lambda item: item[1])
            summary_parts.append(f"Highest {aggregation} **{metric}**: **{label}** ({value:,.2f})")
    answer = "; ".join(summary_parts) + ".\n\n" + _markdown_table(table_rows, [group_by, "records", *metrics], len(table_rows))
    return {
        "answer": answer,
        "sources": [{"file": schema["file"], "text": f"{schema['sheet']} — all rows grouped by {group_by}", "score": 1.0} for schema in schemas],
        "thinking_steps": ["Computed the requested grouping and aggregation from every row in the local structured spreadsheet store."],
    }


def _join_grouped_result(plan: dict, schemas: list[dict], rows: list[dict]) -> dict | None:
    """Join two selected sheets on planned keys, then run the normal group aggregate."""
    left_file, right_file = plan.get("left_file"), plan.get("right_file")
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file"], []).append(row)
    if left_file not in by_file or right_file not in by_file or left_file == right_file:
        return None
    left_rows, right_rows = by_file[left_file], by_file[right_file]
    left_columns = list(left_rows[0]["values"]) if left_rows else []
    right_columns = list(right_rows[0]["values"]) if right_rows else []
    left_key = _resolve_column(plan.get("left_key"), left_columns)
    right_key = _resolve_column(plan.get("right_key"), right_columns)
    if not left_key or not right_key:
        return None
    right_lookup: dict[str, list[dict]] = {}
    for row in right_rows:
        value = row["values"].get(right_key)
        if value not in (None, ""):
            right_lookup.setdefault(str(value).strip(), []).append(row)
    joined = []
    for left in left_rows:
        matches = right_lookup.get(str(left["values"].get(left_key, "")).strip(), [])
        for right in matches:
            joined.append({
                "file": f"{left_file} + {right_file}",
                "sheet": f"{left['sheet']} + {right['sheet']}",
                "row": left["row"],
                "values": {**left["values"], **right["values"]},
            })
    if not joined:
        return None
    result = _grouped_result(plan, schemas, joined)
    if result is not None:
        result["sources"] = [
            {"file": left_file, "text": f"Joined on {left_key}", "score": 1.0},
            {"file": right_file, "text": f"Joined on {right_key}", "score": 1.0},
        ]
        result["thinking_steps"] = [
            f"Joined {left_file}.{left_key} to {right_file}.{right_key}, then aggregated all {len(joined):,} joined rows."
        ]
    return result


def _is_group_request(query: str, columns: list[str]) -> bool:
    """Recognize category/segment questions without naming dataset-specific fields."""
    lowered = query.lower()
    if re.search(r"\b(?:trend|trends|group|groups|across|by)\b", lowered):
        return True
    match = re.search(r"\b(?:which|what)\s+(.+?)\s+(?:has|have|had|shows?|with)\b", lowered)
    return bool(match and _resolve_column(match.group(1), columns))


def _semantic_scope(query: str, plan: dict, schemas: list[dict]) -> dict | None:
    """Select exactly one worksheet/grain for a broad spreadsheet analysis."""
    requested_file = str(plan.get("file") or "").strip().lower()
    requested_sheet = str(plan.get("sheet") or "").strip().lower()
    candidates = [
        schema for schema in schemas
        if (not requested_file or schema["file"].lower() == requested_file)
        and (not requested_sheet or schema["sheet"].lower() == requested_sheet)
    ]
    if requested_file or requested_sheet:
        return candidates[0] if len(candidates) == 1 else None

    files = {schema["file"] for schema in schemas}
    if len(files) != 1:
        return None
    candidates = [schema for schema in schemas if schema["rows"] >= 2 and len(schema["columns"]) >= 2]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(schema: dict):
        name_tokens = set(re.findall(r"[a-z0-9]+", schema["sheet"].lower()))
        value = len(query_tokens & name_tokens) * 4
        if name_tokens & {"us", "national", "overall", "summary"}:
            value += 12
        if name_tokens & {"total", "totals", "aggregate", "aggregated"}:
            value += 8
        if "monthly" in name_tokens and query_tokens & {"trend", "trends", "pattern", "patterns"}:
            value += 6
        if "current" in name_tokens and query_tokens & {"trend", "trends"}:
            value -= 8
        # For an unspecified workbook-wide summary, prefer an already
        # aggregated table over a detailed transactional sheet.
        value -= min(schema["rows"], 10_000) / 10_000
        return value

    return max(candidates, key=score)


def _semantic_dataset_answer(query: str, plan: dict, schemas: list[dict], rows: list[dict]) -> dict | None:
    """Answer broad spreadsheet questions from complete computed evidence, not samples."""
    scope = _semantic_scope(query, plan, schemas)
    if scope is None:
        return None
    scoped_schemas = [scope]
    scoped_rows = [
        row for row in rows
        if row["file"] == scope["file"] and row["sheet"] == scope["sheet"]
    ]
    if not scoped_rows:
        return None
    all_columns = scope["columns"]
    requested = plan.get("relevant_columns") or all_columns
    columns = list(dict.fromkeys(_resolve_column(column, all_columns) for column in requested))
    columns = [column for column in columns if column]
    if not columns:
        return None

    evidence = {
        "file": scope["file"],
        "sheet": scope["sheet"],
        "row_count": len(scoped_rows),
        "columns": {},
    }
    mixed_columns = []
    for column in columns:
        values = [row["values"].get(column) for row in scoped_rows if column in row["values"]]
        non_empty = [value for value in values if not _is_missing(value)]
        numeric = [_number(value) for value in non_empty]
        numeric = [value for value in numeric if value is not None]
        column_evidence = {"non_empty": len(non_empty), "missing": len(values) - len(non_empty)}
        numeric_ratio = len(numeric) / len(non_empty) if non_empty else 0
        # A substantial mix of numeric values and text labels in one field is
        # usually a report layout with embedded subheaders, not a clean table.
        if 0.05 <= numeric_ratio <= 0.95:
            mixed_columns.append(column)
            column_evidence["mixed_data_types"] = True
        if non_empty and len(numeric) / len(non_empty) >= 0.9:
            if min(numeric) == max(numeric):
                column_evidence["constant_value"] = str(numeric[0])
            else:
                column_evidence["numeric"] = {
                    "min": str(min(numeric)), "max": str(max(numeric)),
                    "average": str(sum(numeric) / len(numeric)),
                    "negative_count": sum(value < 0 for value in numeric),
                }
        else:
            counts = Counter(str(value) for value in non_empty)
            column_evidence["top_values"] = [
                {"value": value, "count": count, "percent": round(count * 100 / len(non_empty), 2)}
                for value, count in counts.most_common(12)
            ]
        evidence["columns"][column] = column_evidence

    if mixed_columns:
        return {
            "answer": (
                "Analysis paused because this worksheet is not yet a clean flat table. "
                f"The following field(s) mix numeric values with text/report labels: **{', '.join(mixed_columns)}**. "
                "This usually means the sheet contains embedded subheaders or multiple table regions. "
                "Extracting those regions into separate tables is required before a trustworthy trend summary can be calculated."
            ),
            "sources": [{"file": scope["file"], "text": f"{scope['sheet']} schema validation detected mixed-type fields", "score": 1.0}],
            "thinking_steps": ["Stopped full-dataset analysis to avoid treating report labels as data values."],
        }

    group_by = _resolve_column(plan.get("group_by"), all_columns)
    if group_by:
        group_plan = {
            "group_by": group_by,
            "metrics": [column for column in columns if column != group_by],
            "aggregation": plan.get("aggregation") or "average",
            "bin_size": plan.get("bin_size"),
        }
        grouped = _grouped_result(group_plan, scoped_schemas, scoped_rows)
        if grouped:
            evidence["grouped_analysis"] = grouped["answer"]

    prompt = f"""Answer the spreadsheet question using only this computed evidence.
The evidence was calculated from every relevant row, not a sample. Be clear about
whether figures are averages, totals, counts, or reporting coverage.

Rules:
- The evidence scope is exactly one worksheet. Name that worksheet and do not
  imply that other worksheets were combined.
- Never add a stock/snapshot measure (for example capacity, balance, inventory,
  headcount, or active customers) across months unless the question explicitly
  requests such a sum. Use monthly changes, minimum, maximum, and latest values.
- Missing values show incomplete reporting; never infer low adoption, low use,
  causation, or real-world absence from missing values.
- A frequent category means frequent reporting/records, not market share or
  real-world prevalence, unless the evidence explicitly provides that measure.
- If a numeric metric has negative values where negatives may be unexpected,
  flag them as values requiring validation; do not treat them as ordinary facts.
- Explain meaningful differences between total row counts and non-empty counts
  as missing values for that field.
- A constant date/year field means the data is a snapshot for that period, not
  a time series. Do not call cross-sectional patterns a time trend.
- Do not report redundant average/minimum/maximum statistics for a constant field.
- State only insights supported by the evidence. Do not list hypothetical
  "missing calculations" when the question asks for a summary.

Question: {query}
Computed evidence: {json.dumps(evidence, ensure_ascii=False)}
"""
    try:
        answer = str(_llm().complete(prompt)).strip()
    except Exception:
        return None
    return {
        "answer": answer,
        "sources": [{"file": scope["file"], "text": f"{scope['sheet']} complete structured analysis", "score": 1.0}],
        "thinking_steps": [f"Selected worksheet {scope['sheet']} and computed evidence from all {len(scoped_rows):,} rows at that single grain."],
    }


def _column_names_result(query: str, schemas: list[dict]) -> dict | None:
    """Return spreadsheet headers directly instead of asking the LLM to plan rows."""
    lowered = query.lower()
    asks_for_columns = re.search(
        r"^\s*(?:please\s+)?(?:print|show|list|display|give)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:column|columns|headers?|fields?)(?:\s+names?)?\b"
        r"|^\s*(?:what|which)\s+(?:are|is)\s+(?:all\s+)?(?:the\s+)?(?:column|columns|headers?|fields?)(?:\s+names?)?\b",
        lowered,
    )
    if not asks_for_columns:
        return None

    sections = []
    sources = []
    for schema in schemas:
        columns = schema["columns"]
        sections.append(
            f"**{schema['file']}** ({schema['sheet']}, {len(columns)} columns):\n\n"
            + "\n".join(f"{index}. `{column}`" for index, column in enumerate(columns, start=1))
        )
        sources.append({"file": schema["file"], "text": f"{schema['sheet']} schema", "score": 1.0})
    return {
        "answer": "\n\n".join(sections),
        "sources": sources,
        "thinking_steps": ["Read the complete spreadsheet schema directly from the local structured store."],
    }


def try_structured_query(query: str, filenames: list[str], upload_dir: Path) -> dict | None:
    """Return an exact SQLite-backed result, or None for semantic questions."""
    schemas, rows = structured_snapshot(filenames, upload_dir)
    if not schemas:
        return None
    column_names_result = _column_names_result(query, schemas)
    if column_names_result is not None:
        return column_names_result
    all_columns = list(dict.fromkeys(column for schema in schemas for column in schema["columns"]))
    if len(schemas) == 1 and not _is_group_request(query, all_columns):
        deterministic_result = _deterministic_extreme(query, schemas, rows)
        if deterministic_result is not None:
            return deterministic_result
    schema_text = json.dumps(schemas, ensure_ascii=False)
    prompt = f"""Classify and plan this spreadsheet question using only the supplied schema.
Return JSON only. Use mode "structured" for exact row lookup, filtering, counting,
sum, average, minimum, maximum, or grouping/aggregation across a column. Use mode
"semantic" for broad interpretations that need complete-dataset statistics rather
than a single exact calculation.
Questions such as "which gender has the highest debt", "by region", "monthly trend",
or "trends across age groups" are structured: plan a group operation rather than
sampling rows or answering from a workbook profile.

For structured mode return:
{{"mode":"structured","operation":"rows|count|sum|average|min|max","value_column":null,
 "filters":[{{"column":"exact schema column","operator":"eq|ne|contains|gt|gte|lt|lte","value":"..."}}],
 "sheet":null,"limit":20}}
For grouping return:
{{"mode":"structured","operation":"group","group_by":"exact schema column",
 "metrics":["one or more exact numeric schema columns"],
 "aggregation":"average|sum|min|max|count","bin_size":null,"sheet":null}}
Use bin_size only when the question asks for numeric ranges/groups (for example,
10 for age groups). Use average for a phrase such as "which gender has the
highest debt" unless the question explicitly asks for total, minimum, or maximum.
If the grouping field and metric are in different selected sheets, return:
{{"mode":"structured","operation":"join_group","left_file":"exact file name",
 "right_file":"exact file name","left_key":"exact left column","right_key":"exact right column",
 "group_by":"exact column after joining","metrics":["exact numeric column after joining"],
 "aggregation":"average|sum|min|max|count","bin_size":null}}
For semantic analysis return:
{{"mode":"semantic","file":"exact file name","sheet":"exact worksheet name",
 "relevant_columns":["exact schema columns needed"],
 "group_by":"exact schema column or null","aggregation":"average|sum|count",
 "bin_size":null}}
Use semantic analysis for broad questions such as trends, patterns, comparisons,
or themes. Select only the fields needed for the question; the system will compute
evidence from every relevant row before asking the LLM to explain it.
When a workbook has multiple sheets, select exactly one sheet whose grain matches
the question. Prefer a national/overall totals time-series sheet for an unspecified
workbook-wide trend summary; use state or utility sheets only when the question
asks for those dimensions. Never merge sheets with different grains.
For a trend/pattern question, set group_by to the exact date/month/year column when
the selected sheet contains one. For stock/snapshot metrics such as capacity,
balance, inventory, or customer count, use average by period rather than summing
the values across periods.
Never invent a column. For count, value_column must be null.

Schema: {schema_text}
Question: {query}
"""
    try:
        plan = _json_object(str(_llm().complete(prompt)))
    except Exception:
        return None
    if plan.get("mode") == "semantic":
        return _semantic_dataset_answer(query, plan, schemas, rows)
    if plan.get("mode") != "structured" or plan.get("operation") not in _OPERATIONS:
        return None
    sheet = plan.get("sheet")
    if plan["operation"] != "join_group" and len(schemas) > 1 and not sheet:
        return {
            "answer": (
                "This workbook contains multiple worksheets at different levels of detail. "
                "The requested calculation did not identify one worksheet, so it was stopped "
                "to prevent double-counting. Please name the worksheet or ask for a workbook-wide summary."
            ),
            "sources": [],
            "thinking_steps": ["Stopped an ambiguous cross-sheet calculation before combining incompatible row grains."],
        }
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

    if plan["operation"] == "group":
        return _grouped_result(plan, schemas, candidates)
    if plan["operation"] == "join_group":
        return _join_grouped_result(plan, schemas, candidates)

    operation = plan["operation"]
    sources = [{"file": row["file"], "text": f"{row['sheet']} row {row['row']}", "score": 1.0} for row in candidates[:8]]
    if operation == "rows":
        limit = max(1, min(int(plan.get("limit") or 20), 50))
        answer = f"Found **{len(candidates)} matching row(s)**.\n\n" + _markdown_table(candidates, all_columns, limit)
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
