"""
Smart query router: explicit-mode constraints first, then the configured Groq LLM.
Returns the chosen approach name + human-readable reasoning.
"""

import os
from groq import Groq

from config import Config
from utils import classify_files
from followup import explicit_engine_request


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

- basic_rag       : Specific factual, numerical, table, section, or equation lookup that can be answered from directly retrieved passages. It may ask for more than one closely related fact from the same context.
- subquestion     : A question that genuinely needs decomposition into independent sub-questions, synthesis of distinct evidence, or comparison across documents.
- router_engine   : A broad summary, overview, main-theme question, or a question where the system should choose between whole-document summary and targeted retrieval.

Rules:
- Decide from the meaning and information needs of the complete query, not from one keyword.
- Multiple requested facts do not automatically require subquestion if direct retrieval can answer them together.
- The number of uploaded documents is context, not a deterministic rule.
- Prefer basic_rag for exact values and explicitly labelled document elements.
- Prefer router_engine only when broad document-level understanding or summary selection is useful.
- Prefer subquestion only when decomposition and later synthesis materially improve the answer.

Output ONLY the single label. No explanation."""


class QueryRouter:
    def __init__(self):
        self._client = None

    def _groq(self):
        if self._client is None:
            self._client = Groq(api_key=Config.GROQ_API_KEY)
        return self._client

    def _llm_classify(self, query: str, num_text_files: int) -> tuple[str, str | None]:
        """Call Groq to classify the query. Falls back to basic_rag on any error.

        Returns (label, error_msg). error_msg is None on success, the exception
        string on failure — so callers can show an honest reason instead of
        claiming the model 'classified' something it never saw.
        """
        try:
            user_msg = f"Number of uploaded text documents: {num_text_files}\nQuery: {query}"
            resp = self._groq().chat.completions.create(
                model=Config.GROQ_ROUTER_LLM,
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
            return label, None
        except Exception as e:
            err = str(e)
            print(f"[Router] LLM classification failed ({err}), defaulting to basic_rag")
            return "basic_rag", err

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

        requested_engine = explicit_engine_request(query)
        text_engines = {"basic_rag", "router_engine", "subquestion", "react", "multi_document"}
        request_is_compatible = (
            requested_engine == "multimodal" and has_images and not has_texts
        ) or (
            requested_engine in text_engines and has_texts and not has_images
        )
        if requested_engine and request_is_compatible:
            return {
                "label": requested_engine,
                "approach": APPROACH_LABELS[requested_engine],
                "reason": f"User explicitly requested the {APPROACH_LABELS[requested_engine]} for this query.",
            }

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

        # ── LLM classification for normal text-only queries ───────────────────
        label, classify_err = self._llm_classify(query, len(texts))

        if classify_err:
            # LLM call failed — be honest about what happened instead of claiming
            # the model made a classification decision it never actually made.
            reason = (
                f"Router LLM ({Config.GROQ_ROUTER_LLM}) failed — defaulting to Basic RAG. "
                f"Error: {classify_err}"
            )
        else:
            reasons = {
                "basic_rag": f"Groq/{Config.GROQ_ROUTER_LLM} classified this as a targeted retrieval task → Basic RAG.",
                "subquestion": f"Groq/{Config.GROQ_ROUTER_LLM} detected multi-part or cross-document comparison across {len(texts)} docs → Sub-Question Engine.",
                "router_engine": f"Groq/{Config.GROQ_ROUTER_LLM} detected summarization or broad overview query → Router Query Engine.",
            }
            reason = reasons[label]

        return {
            "label": label,
            "approach": APPROACH_LABELS[label],
            "reason": reason,
        }
