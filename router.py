"""
Smart query router: explicit-mode constraints first, then the configured Groq LLM.
Returns the chosen approach name + human-readable reasoning.
"""

import re
from llama_index.llms.google_genai import GoogleGenAI

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

_QUESTION_INTENT_RE = re.compile(
    r"\b(?:what|which|who|when|where|why|how|compare|contrast|explain|summari[sz]e|list|describe|identify)\b",
    re.IGNORECASE,
)
_BROAD_QUERY_RE = re.compile(
    r"\b(?:summari[sz]e|overview|main\s+(?:topic|theme|idea)|key\s+(?:points|takeaways)|whole\s+document|document\s+as\s+a\s+whole)\b",
    re.IGNORECASE,
)


def _is_multi_part_multi_document_query(query: str, num_text_files: int) -> bool:
    """Recognize independently answerable requests before calling the router LLM."""
    return num_text_files >= 2 and len(_QUESTION_INTENT_RE.findall(query)) >= 2

ROUTER_SYSTEM_PROMPT = """You are a query routing assistant for a document QA system.
Given a user query and context, output ONLY one label from this list:

- basic_rag       : Specific factual, numerical, table, section, or equation lookup that can be answered from directly retrieved passages. It may ask for more than one closely related fact from the same context.
- subquestion     : A question that genuinely needs decomposition into independent sub-questions, synthesis of distinct evidence, or comparison across documents. Use this when the user asks several independent things (for example embeddings, ranking method, and paper formulas), especially with two or more documents.
- router_engine   : A broad summary, overview, main-theme question, or a question where the system should choose between whole-document summary and targeted retrieval.

Rules:
- Decide from the meaning and information needs of the complete query, not from one keyword.
- Keep basic_rag only when all requested facts can be answered from the same small set of passages. Route to subquestion when requests concern distinct concepts, sections, methods, or documents and each needs separate evidence.
- The number of uploaded documents is context, not a deterministic rule.
- Prefer basic_rag for exact values and explicitly labelled document elements.
- Prefer router_engine only when broad document-level understanding or summary selection is useful.
- Prefer subquestion only when decomposition and later synthesis materially improve the answer.

Output ONLY the single label. No explanation."""


class QueryRouter:
    def __init__(self):
        self._client = None

    def _gemini(self):
        if self._client is None:
            self._client = GoogleGenAI(
                api_key=Config.GOOGLE_API_KEY,
                model=Config.GOOGLE_LLM,
                temperature=0,
                max_tokens=10,
                max_retries=Config.GOOGLE_MAX_RETRIES,
                is_function_calling_model=False,
            )
        return self._client

    def _llm_classify(self, query: str, num_text_files: int) -> tuple[str, str | None]:
        """Call Gemini to classify the query. Falls back to basic_rag on any error.

        Returns (label, error_msg). error_msg is None on success, the exception
        string on failure — so callers can show an honest reason instead of
        claiming the model 'classified' something it never saw.
        """
        try:
            prompt = (
                f"{ROUTER_SYSTEM_PROMPT}\n\n"
                f"Number of uploaded text documents: {num_text_files}\n"
                f"Query: {query}"
            )
            label = str(self._gemini().complete(prompt)).strip().lower()
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
        if len(texts) >= 2 and all(name.lower().endswith((".csv", ".xlsx")) for name in texts):
            spreadsheet_label, spreadsheet_err = self._llm_classify(query, len(texts))
            if not spreadsheet_err and spreadsheet_label == "basic_rag":
                return {
                    "label": "basic_rag",
                    "approach": APPROACH_LABELS["basic_rag"],
                    "reason": (
                        f"Gemini/{Config.GOOGLE_LLM} detected an exact structured query "
                        "across the selected spreadsheets -> local SQLite path."
                    ),
                }

        if len(texts) >= 2:
            return {
                "label": "subquestion",
                "approach": APPROACH_LABELS["subquestion"],
                "reason": (
                    "Multiple selected documents detected -> Sub-Question Engine."
                ),
            }

        if _BROAD_QUERY_RE.search(query):
            return {
                "label": "router_engine",
                "approach": APPROACH_LABELS["router_engine"],
                "reason": "Broad single-document question detected -> Router Query Engine.",
            }

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
            reasons = {
                "basic_rag": f"Gemini/{Config.GOOGLE_LLM} classified this as a targeted retrieval task -> Basic RAG.",
                "subquestion": f"Gemini/{Config.GOOGLE_LLM} detected a multi-part or cross-document request -> Sub-Question Engine.",
                "router_engine": f"Gemini/{Config.GOOGLE_LLM} detected a broad summary or overview -> Router Query Engine.",
            }
            reason = reasons[label]

        return {
            "label": label,
            "approach": APPROACH_LABELS[label],
            "reason": reason,
        }
