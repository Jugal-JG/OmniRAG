"""Deterministic query normalization and explicit-intent helpers."""

from __future__ import annotations

import re


_REFERENCE_TYPOS = {
    "forumula": "formula",
    "formual": "formula",
    "fromula": "formula",
    "formla": "formula",
    "fomula": "formula",
    "equaiton": "equation",
    "equaton": "equation",
}
_REFERENCE_WORDS = (
    r"formulas?|equations?|eqn?|tables?|figures?|fig|sections?|sec|theorems?|"
    r"lemmas?|corollar(?:y|ies)|algorithms?|alg|definitions?|def|"
    r"propert(?:y|ies)|chapters?|steps?|lines?|examples?"
)
_PAREN_NUMBER_RE = re.compile(r"\((\d+)\)")
_REFERENCE_LIST_RE = re.compile(
    rf"(?:{_REFERENCE_WORDS})\.?\s*"
    r"((?:\(?\d+\)?)(?:\s*(?:,|and|&)\s*\(?\d+\)?)*)",
    re.IGNORECASE,
)

_LABELLED_REFERENCE_RE = re.compile(
    r"\b(formulas?|equations?|eq\.?|tables?|figures?|fig\.?|sections?|sec\.?|theorems?|"
    r"lemma|algorithm|definition)\s*(?:label(?:led)?\s*)?\(?\s*(\d+)\s*\)?",
    re.IGNORECASE,
)
_VAGUE_REFERENCE_RE = re.compile(
    r"\b(?:the|this|that)\s+(?:formulas?|equations?|tables?|figures?|sections?|theorems?|"
    r"lemma|algorithm|definition)\b|\b(?:it|its|this|that)\b",
    re.IGNORECASE,
)
_REFERENCE_NOUN_RE = re.compile(
    r"\b(formulas?|equations?|tables?|figures?|sections?|theorems?|lemmas?|algorithms?|definitions?)\b",
    re.IGNORECASE,
)
_ENGINE_REQUEST_PREFIX = r"\b(?:use|using|with|via|choose|select|route(?:\s+this\s+query)?\s+to)\s+(?:the\s+)?"
_ENGINE_REQUEST_PATTERNS = (
    ("basic_rag", re.compile(_ENGINE_REQUEST_PREFIX + r"basic[\s-]*rag(?:\s+engine)?\b", re.IGNORECASE)),
    ("router_engine", re.compile(_ENGINE_REQUEST_PREFIX + r"router(?:\s+query)?\s+engine\b", re.IGNORECASE)),
    ("subquestion", re.compile(_ENGINE_REQUEST_PREFIX + r"sub[\s-]*question(?:\s+engine)?\b", re.IGNORECASE)),
    ("react", re.compile(_ENGINE_REQUEST_PREFIX + r"(?:react(?:\s+agent)?|thinking\s+mode)\b", re.IGNORECASE)),
    ("multi_document", re.compile(_ENGINE_REQUEST_PREFIX + r"multi[\s-]*document(?:\s+agent|\s+engine)?\b", re.IGNORECASE)),
    ("multimodal", re.compile(_ENGINE_REQUEST_PREFIX + r"multi[\s-]*modal(?:\s+engine)?\b", re.IGNORECASE)),
)


def _normalise_kind(kind: str) -> str:
    kind = kind.lower().rstrip(".")
    return {
        "eq": "equation",
        "fig": "figure",
        "sec": "section",
        "formulas": "formula",
        "equations": "equation",
        "tables": "table",
        "figures": "figure",
        "sections": "section",
        "theorems": "theorem",
        "lemmas": "lemma",
        "algorithms": "algorithm",
        "definitions": "definition",
    }.get(kind, kind)


def normalize_reference_typos(query: str) -> str:
    """Correct common reference-word typos without modifying document content."""
    result = query
    for typo, replacement in _REFERENCE_TYPOS.items():
        result = re.sub(rf"\b{re.escape(typo)}\b", replacement, result, flags=re.IGNORECASE)
    return result


def extract_labelled_reference_numbers(query: str) -> set[str]:
    """Extract all labels from queries such as 'formulas 15 and 16'."""
    query = normalize_reference_typos(query)
    numbers = set(_PAREN_NUMBER_RE.findall(query))
    for match in _REFERENCE_LIST_RE.finditer(query):
        numbers.update(re.findall(r"\d+", match.group(1)))
    return numbers


def explicit_engine_request(query: str) -> str | None:
    """Return an engine explicitly requested in natural-language query text."""
    for engine, pattern in _ENGINE_REQUEST_PATTERNS:
        if pattern.search(query):
            return engine
    return None


def _find_recent_reference(history: list[dict]) -> tuple[str, str] | None:
    """Prefer explicit labels in user questions, then explicit labels in answers."""
    for field in ("q", "a"):
        for turn in reversed(history):
            match = _LABELLED_REFERENCE_RE.search(str(turn.get(field, "")))
            if match:
                return _normalise_kind(match.group(1)), match.group(2)
    return None


def resolve_labelled_followup(query: str, history: list[dict]) -> str:
    """Carry the most recent formula/table/etc. label into a vague follow-up."""
    query = normalize_reference_typos(query)
    if not history or _LABELLED_REFERENCE_RE.search(query):
        return query
    if not (_VAGUE_REFERENCE_RE.search(query) or _REFERENCE_NOUN_RE.search(query)):
        return query

    reference = _find_recent_reference(history)
    if not reference:
        return query
    kind, number = reference

    noun_match = _REFERENCE_NOUN_RE.search(query)
    if noun_match:
        start, end = noun_match.span()
        return f"{query[:start]}{kind} ({number}){query[end:]}"

    return f"{query} The referenced {kind} is {kind} ({number})."
