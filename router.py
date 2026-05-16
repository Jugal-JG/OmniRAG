"""
Smart query router: rule-based first, then Groq/Llama-4-scout for ambiguous queries.
Returns the chosen approach name + human-readable reasoning.
"""

import os
from groq import Groq

from config import Config
from utils import classify_files


APPROACH_LABELS = {
    "basic_rag": "Basic RAG",
    "multi_document": "Multi-Document Agent",
    "multimodal": "Multi-Modal",
    "react": "ReAct Agent",
    "router_engine": "Router Query Engine",
    "subquestion": "Sub-Question Engine",
    "merged": "Merged (Multi-Modal + Text)",
}

ROUTER_SYSTEM_PROMPT = """You are a query routing assistant for a document QA system.
Given a user query and context, output ONLY one label from this list:

- basic_rag       : Simple, narrow factual question from a single document.
- subquestion     : Multi-part question, comparison across documents, or question needing decomposition.
- router_engine   : Summarization / overview / "what is this about" queries, or when both summary and specific search may be needed.

Rules:
- If the query contains words like "compare", "difference between", "vs", "both", "across" → subquestion
- If the query contains "summarize", "overview", "what is this about", "explain briefly" → router_engine
- Default to basic_rag for specific factual lookups.

Output ONLY the single label. No explanation."""


class QueryRouter:
    def __init__(self):
        self._client = None

    def _groq(self):
        if self._client is None:
            self._client = Groq(api_key=Config.GROQ_API_KEY)
        return self._client

    def _llm_classify(self, query: str, num_text_files: int) -> str:
        """Call Groq to classify the query. Falls back to basic_rag on any error."""
        try:
            user_msg = f"Number of uploaded text documents: {num_text_files}\nQuery: {query}"
            resp = self._groq().chat.completions.create(
                model=Config.GROQ_LLM,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=10,
                temperature=0,
            )
            label = resp.choices[0].message.content.strip().lower()
            if label not in ("basic_rag", "subquestion", "router_engine"):
                label = "basic_rag"
            return label
        except Exception as e:
            print(f"[Router] LLM classification failed ({e}), defaulting to basic_rag")
            return "basic_rag"

    def route(
        self,
        query: str,
        filenames: list[str],
        multi_doc_mode: bool = False,
        thinking_mode: bool = False,
    ) -> dict:
        """
        Returns {"approach": str, "reason": str, "label": str}
        """
        images, texts = classify_files(filenames)
        has_images = bool(images)
        has_texts = bool(texts)

        # ── Rule-based (no API call) ──────────────────────────────────────────
        if thinking_mode:
            return {
                "label": "react",
                "approach": APPROACH_LABELS["react"],
                "reason": "ReAct thinking mode enabled — using step-by-step tool reasoning.",
            }

        if multi_doc_mode:
            return {
                "label": "multi_document",
                "approach": APPROACH_LABELS["multi_document"],
                "reason": "Multi-document agent mode enabled — each uploaded file gets its own agent.",
            }

        if has_images and not has_texts:
            return {
                "label": "multimodal",
                "approach": APPROACH_LABELS["multimodal"],
                "reason": "Only images detected — routing to Groq vision engine.",
            }

        if has_images and has_texts:
            return {
                "label": "merged",
                "approach": APPROACH_LABELS["merged"],
                "reason": "Both images and text documents detected — running multimodal + text engine and merging answers.",
            }

        if not has_texts:
            return {
                "label": "basic_rag",
                "approach": APPROACH_LABELS["basic_rag"],
                "reason": "No documents uploaded yet — using Basic RAG on any pre-loaded data.",
            }

        # ── LLM classification for text-only queries ──────────────────────────
        query_l = query.lower()
        comparison_terms = (
            "compare",
            "comparison",
            "difference",
            "differences",
            "similar",
            "similarities",
            "vs",
            "versus",
            "both",
            "across",
            "between",
            "each",
            "they",
            "their",
            "them",
            "these",
            "all documents",
            "uploaded documents",
        )
        if len(texts) > 1 and any(term in query_l for term in comparison_terms):
            return {
                "label": "subquestion",
                "approach": APPROACH_LABELS["subquestion"],
                "reason": (
                    "Rule detected a cross-document or plural question across "
                    f"{len(texts)} docs â€” Sub-Question Engine."
                ),
            }

        label = self._llm_classify(query, len(texts))

        # Sub-Question Engine is for comparing MULTIPLE documents.
        # With only 1 text file it decomposes unnecessarily and misses direct facts —
        # downgrade to Router Engine which handles both summary and vector search.
        if label == "subquestion" and len(texts) == 1:
            label = "router_engine"
            return {
                "label": label,
                "approach": APPROACH_LABELS[label],
                "reason": (
                    "Query looked multi-part, but only 1 document is uploaded — "
                    "using Router Engine (summary + vector) instead of Sub-Question Engine."
                ),
            }

        reasons = {
            "basic_rag": "Groq/Llama-4-scout classified this as a narrow factual lookup → Basic RAG.",
            "subquestion": f"Groq/Llama-4-scout detected multi-part or cross-document comparison across {len(texts)} docs → Sub-Question Engine.",
            "router_engine": "Groq/Llama-4-scout detected summarization or broad overview query → Router Query Engine.",
        }
        return {
            "label": label,
            "approach": APPROACH_LABELS[label],
            "reason": reasons[label],
        }
