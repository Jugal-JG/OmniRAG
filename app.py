"""Flask application: LlamaIndex Multi-Engine Explorer."""

# ── NOTE: nest_asyncio removed ────────────────────────────────────────────────
# Python 3.14 no longer allows nest_asyncio to monkey-patch asyncio.
# Agent engines (multi_document, react_agent) now run async code in dedicated
# threads via concurrent.futures + asyncio.run(), which provides a clean
# event loop without needing nest_asyncio.

import json
import logging
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
    history = _load_history(_session_dir())
    if not history:
        return query_text

    last_turns = history[-MAX_HISTORY:]
    history_text = "\n".join(
        f"User: {t['q']}\nAssistant: {t['a']}" for t in last_turns
    )
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {query_text}\n\n"
        "Rewrite the follow-up as a single, fully self-contained question that includes "
        "all context needed from the conversation above. "
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
    filenames = _uploaded_files(upload_dir)

    filenames = [f for f in filenames if (upload_dir / f).exists()]

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
    routing = router.route(
        query=query_text,
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

            text_label = router.route(
                query=query_text, filenames=texts, multi_doc_mode=False, thinking_mode=False
            )["label"]
            text_engine = get_engine(text_label)
            text_result = text_engine(standalone_query, texts, upload_dir)

            from groq import Groq

            client = Groq(api_key=Config.GROQ_API_KEY)
            merge_prompt = (
                f"Image analysis result:\n{mm_result['answer']}\n\n"
                f"Text document analysis result:\n{text_result['answer']}\n\n"
                f"Original question: {query_text}\n\n"
                "Synthesize both analyses into one comprehensive, clear answer."
            )
            merge_resp = client.chat.completions.create(
                model=Config.GROQ_LLM,
                messages=[{"role": "user", "content": merge_prompt}],
                max_tokens=1024,
            )
            merged_answer = merge_resp.choices[0].message.content

            result = {
                "answer": merged_answer,
                "sources": text_result.get("sources", []),
                "thinking_steps": [
                    f"[Image analysis] {mm_result['answer'][:300]}…",
                    f"[Text analysis ({text_label})] {text_result['answer'][:300]}…",
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

    answer = result.get("answer", "")
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
