"""Flask application: LlamaIndex Multi-Engine Explorer."""

# ── NOTE: nest_asyncio removed ────────────────────────────────────────────────
# Python 3.14 no longer allows nest_asyncio to monkey-patch asyncio.
# Agent engines (multi_document, react_agent) now run async code in dedicated
# threads via concurrent.futures + asyncio.run(), which provides a clean
# event loop without needing nest_asyncio.

import json
import logging
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from engines import get_engine
from engines.multimodal import run as multimodal_run
from router import QueryRouter
from utils import classify_files
from answer_format import MATH_FORMAT_INSTRUCTIONS
from followup import (
    explicit_engine_request,
    normalize_reference_typos,
    resolve_labelled_followup,
)
from answer_format import repair_bare_latex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="frontend", static_folder="frontend/static")
app.secret_key = Config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["SESSION_COOKIE_SAMESITE"] = Config.SESSION_COOKIE_SAMESITE
app.config["SESSION_COOKIE_SECURE"] = Config.SESSION_COOKIE_SECURE
CORS(app, origins=Config.CORS_ORIGINS, supports_credentials=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".html", ".md"}
MAX_HISTORY = 3

Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(Config.CACHE_FOLDER).mkdir(parents=True, exist_ok=True)

router = QueryRouter()


def _preload_embedding_model() -> None:
    """Load embeddings during both Flask and Gunicorn application startup."""
    if not Config.PRELOAD_EMBED_MODEL:
        return
    import model_cache

    logger.info("[startup] Pre-loading embedding model: %s", Config.EMBED_MODEL)
    model_cache.get_embed_model(Config.EMBED_MODEL)
    logger.info("[startup] Embedding model ready.")


_preload_embedding_model()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _client_session_id() -> str:
    raw_sid = request.headers.get("X-Omnirag-Session-Id") or session.get("session_id")
    try:
        sid = str(uuid.UUID(str(raw_sid)))
    except (TypeError, ValueError):
        sid = str(uuid.uuid4())
    session["session_id"] = sid
    return sid


def _session_dir() -> Path:
    sid = _client_session_id()
    d = Path(Config.UPLOAD_FOLDER) / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _uploaded_files(upload_dir: Path) -> list[str]:
    return sorted(
        p.name for p in upload_dir.iterdir() if p.is_file() and allowed_file(p.name)
    )


def _looks_like_scanned_pdf(file_path: Path) -> bool:
    if file_path.suffix.lower() != ".pdf":
        return False

    try:
        import pypdf

        reader = pypdf.PdfReader(str(file_path))
        extracted = []
        for page in reader.pages[: min(2, len(reader.pages))]:
            extracted.append(page.extract_text() or "")
        return len(" ".join(extracted).strip()) < 40
    except Exception as exc:
        logger.info("[upload] PDF scan check skipped for %s (%s)", file_path.name, exc)
        return False



def _history_file(upload_dir: Path) -> Path:
    return upload_dir / ".history.json"


def _load_history(upload_dir: Path) -> list[dict]:
    try:
        with _history_file(upload_dir).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_history_file(upload_dir: Path, history: list[dict]) -> None:
    with _history_file(upload_dir).open("w", encoding="utf-8") as f:
        json.dump(history[-20:], f)


def _make_standalone(query_text: str) -> str:
    """
    If conversation history exists, use Groq to rewrite the follow-up as a
    fully self-contained question so similarity search isn't polluted by history.
    """
    query_text = normalize_reference_typos(query_text)
    history = _load_history(_session_dir())
    if not history:
        return query_text

    deterministic = resolve_labelled_followup(query_text, history)
    if deterministic != query_text:
        logger.info(
            "[standalone-question] Resolved labelled follow-up deterministically: %r",
            deterministic,
        )
        return deterministic

    last_turns = history[-MAX_HISTORY:]
    history_text = "\n".join(
        f"User: {t['q']}\nAssistant: {t['a']}" for t in last_turns
    )
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {query_text}\n\n"
        "Rewrite the follow-up as a single, fully self-contained question that includes "
        "all context needed from the conversation above. "
        "Preserve and repeat exact document reference labels such as formula (15), "
        "Table 3, or Section 4.2 whenever the follow-up refers to them indirectly. "
        "If the question is already self-contained, return it unchanged. "
        "Output ONLY the rewritten question — no explanation, no quotes."
    )
    try:
        from groq import Groq

        client = Groq(api_key=Config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=Config.GROQ_LLM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        reformulated = resp.choices[0].message.content.strip()
        # Strip <think>...</think> block if present (e.g. from Qwen reasoning model)
        import re
        if "<think>" in reformulated:
            reformulated = re.sub(r"<think>.*?</think>", "", reformulated, flags=re.DOTALL).strip()
            # If the closing tag was cut off due to max_tokens:
            if "<think>" in reformulated:
                reformulated = reformulated.split("<think>")[0].strip()
        return reformulated if reformulated else query_text
    except Exception as e:
        logger.info("[standalone-question] Groq reformulation failed (%s), using original query", e)
        return query_text


def _save_to_history(q: str, a: str, approach: str):
    upload_dir = _session_dir()
    history = _load_history(upload_dir)
    history.append({"q": q, "a": a, "approach": approach})
    _save_history_file(upload_dir, history)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/config.js")
def config_js():
    return send_from_directory("frontend", "config.js", mimetype="application/javascript")


@app.route("/")
def index():
    host = request.host.split(":", 1)[0]
    if host in {"127.0.0.1", "localhost"}:
        return render_template("index.html", missing_keys=Config.missing_keys())

    return jsonify(
        {
            "ok": True,
            "service": "OmniRAG backend",
            "version": Config.APP_VERSION,
            "frontend": "Deploy app/frontend on Vercel and set OMNIRAG_API_BASE_URL to this Space URL.",
        }
    )



def _preindex_basic_rag(filenames: list[str], upload_dir: Path):
    """Pre-build the Basic RAG vector index in a background thread.

    Called right after file upload so the index is ready before the user
    types their first query.  Errors are logged but never surface to the
    user — worst case the index is built lazily on first query as before.
    """
    try:
        from engines.basic_rag import _build_or_load_index

        import model_cache
        from llama_index.core import Settings

        embed_model = model_cache.get_embed_model(Config.EMBED_MODEL)
        Settings.embed_model = embed_model
        Settings.chunk_size = Config.CHUNK_SIZE
        Settings.chunk_overlap = Config.CHUNK_OVERLAP

        logger.info("[preindex] Building Basic RAG index for %s", filenames)
        _build_or_load_index(filenames, upload_dir)
        logger.info("[preindex] Basic RAG index ready for %s", filenames)
    except Exception as exc:
        logger.warning("[preindex] Background indexing failed (%s), will retry on first query", exc)


@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = _session_dir()
    uploaded = []
    errors = []
    warnings = []

    for f in request.files.getlist("files"):
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            errors.append(f"{f.filename} — unsupported file type")
            continue
        filename = secure_filename(f.filename)
        file_path = upload_dir / filename
        f.save(str(file_path))
        uploaded.append(filename)
        if _looks_like_scanned_pdf(file_path):
            warnings.append(
                f"{filename} looks image-based/scanned, so answers may take longer."
            )

    existing = _uploaded_files(upload_dir)
    combined = list(dict.fromkeys(existing + uploaded))

    # Clear conversation history when files change — stale context from
    # previous file sets can confuse the standalone-query reformulator.
    if uploaded:
        _save_history_file(upload_dir, [])

    # Pre-build the Basic RAG index in the background so the first query
    # is instant.  We only index text files (not images).
    text_files = [f for f in combined if not f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    if text_files:
        threading.Thread(
            target=_preindex_basic_rag,
            args=(text_files, upload_dir),
            daemon=True,
        ).start()

    return jsonify(
        {"success": True, "files": combined, "errors": errors, "warnings": warnings}
    )


@app.route("/remove-file", methods=["POST"])
def remove_file():
    data = request.get_json(force=True)
    fname = data.get("filename", "")
    upload_dir = _session_dir()
    filenames = _uploaded_files(upload_dir)

    if fname in filenames:
        filenames.remove(fname)
        try:
            (upload_dir / fname).unlink(missing_ok=True)
        except Exception:
            pass
    # Clear conversation history when files change
    _save_history_file(upload_dir, [])

    return jsonify({"success": True, "files": filenames})


@app.route("/clear-files", methods=["POST"])
def clear_files():
    upload_dir = _session_dir()
    for f in _uploaded_files(upload_dir):
        try:
            (upload_dir / f).unlink(missing_ok=True)
        except Exception:
            pass
    _save_history_file(upload_dir, [])
    return jsonify({"success": True})


@app.route("/new-chat", methods=["POST"])
def new_chat():
    """Clear conversation history only — keep uploaded files intact."""
    _save_history_file(_session_dir(), [])
    return jsonify({"success": True})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    query_text = (data.get("query") or "").strip()
    multi_doc_mode = bool(data.get("multi_doc"))
    thinking_mode = bool(data.get("thinking"))

    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    upload_dir = _session_dir()
    available_files = _uploaded_files(upload_dir)
    requested_files = data.get("selected_files")
    if requested_files is None:
        # Keep API clients written before document selection was added working.
        filenames = available_files
    elif not isinstance(requested_files, list):
        return jsonify({"error": "selected_files must be a list."}), 400
    else:
        # Only accept names belonging to this browser session.  Never let a
        # client select an arbitrary path on disk.
        requested_names = {name for name in requested_files if isinstance(name, str)}
        filenames = [name for name in available_files if name in requested_names]

    if not filenames:
        return jsonify({"error": "Select at least one uploaded document."}), 400

    images, texts = classify_files(filenames)
    logger.info(
        "[query] files=%s text=%s image=%s multi_doc=%s thinking=%s query=%r",
        len(filenames),
        len(texts),
        len(images),
        multi_doc_mode,
        thinking_mode,
        query_text,
    )

    standalone_query = _make_standalone(query_text)
    standalone_note = (
        f" (reformulated: '{standalone_query}')"
        if standalone_query != query_text
        else ""
    )

    # ── Route ──────────────────────────────────────────────────────────────────
    routing_query = query_text if explicit_engine_request(query_text) else standalone_query
    routing = router.route(
        query=routing_query,
        filenames=filenames,
        multi_doc_mode=multi_doc_mode,
        thinking_mode=thinking_mode,
    )
    label = routing["label"]
    if standalone_note:
        routing["reason"] += standalone_note
    logger.info(
        "[query] route=%s approach=%s reason=%s",
        label,
        routing["approach"],
        routing["reason"],
    )

    # ── Execute ────────────────────────────────────────────────────────────────
    try:
        if label == "merged":
            mm_result = multimodal_run(standalone_query, images, upload_dir)

            text_routing = router.route(
                query=query_text, filenames=texts, multi_doc_mode=False, thinking_mode=False
            )
            text_label = text_routing["label"]
            text_engine = get_engine(text_label)
            text_result = text_engine(standalone_query, texts, upload_dir)

            routing["reason"] = (
                f"{routing['reason']} Images routed to Multimodal engine. "
                f"Text routed to {text_routing['approach']} ({text_routing['reason']})."
            )

            from llama_index.llms.mistralai import MistralAI
            merge_prompt = (
                f"Image analysis result:\n{mm_result['answer']}\n\n"
                f"Text document analysis result:\n{text_result['answer']}\n\n"
                f"Original question: {query_text}\n\n"
                "Synthesize both analyses into one comprehensive, clear answer. "
                "Do not output any internal reasoning, scratchpad, or markdown tags like <think>."
                + MATH_FORMAT_INSTRUCTIONS
            )
            merge_llm = MistralAI(
                api_key=Config.MISTRAL_API_KEY,
                model=Config.MISTRAL_LLM,
                max_tokens=Config.MERGED_ANSWER_MAX_TOKENS,
            )
            merged_answer = str(merge_llm.complete(merge_prompt)).strip()

            # Strip <think>...</think> block if present
            import re
            if "<think>" in merged_answer:
                merged_answer = re.sub(r"<think>.*?</think>", "", merged_answer, flags=re.DOTALL).strip()
                if "<think>" in merged_answer:
                    merged_answer = merged_answer.split("<think>")[0].strip()

            result = {
                "answer": merged_answer,
                "sources": text_result.get("sources", []),
                "thinking_steps": [
                    f"[Image analysis] {mm_result['answer']}",
                    f"[Text analysis ({text_label})] {text_result['answer']}",
                ],
            }
        else:
            engine_fn = get_engine(label)
            effective_files = images if label == "multimodal" else filenames
            logger.info("[query] executing engine=%s files=%s", label, effective_files)
            result = engine_fn(standalone_query, effective_files, upload_dir)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": str(exc),
                    "approach": routing["approach"],
                    "router_reason": routing["reason"],
                }
            ),
            500,
        )

    answer = repair_bare_latex(result.get("answer", ""))
    result["answer"] = answer
    logger.info("[query] completed engine=%s answer_chars=%s", label, len(answer))
    _save_to_history(query_text, answer[:600], routing["approach"])

    return jsonify(
        {
            "approach": routing["approach"],
            "approach_label": label,
            "router_reason": routing["reason"],
            "answer": answer,
            "thinking_steps": result.get("thinking_steps", []),
            "sources": result.get("sources", []),
        }
    )


@app.route("/api-status")
def api_status():
    return jsonify(
        {
            "mistral": bool(Config.MISTRAL_API_KEY),
            "google": bool(Config.GOOGLE_API_KEY),
            "groq": bool(Config.GROQ_API_KEY),
            "cohere": bool(Config.COHERE_API_KEY),
            "version": Config.APP_VERSION,
        }
    )


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "version": Config.APP_VERSION})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
