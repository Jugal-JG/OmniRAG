"""Flask application: LlamaIndex Multi-Engine Explorer."""

# ── NOTE: nest_asyncio removed ────────────────────────────────────────────────
# Python 3.14 no longer allows nest_asyncio to monkey-patch asyncio.
# Agent engines (multi_document, react_agent) now run async code in dedicated
# threads via concurrent.futures + asyncio.run(), which provides a clean
# event loop without needing nest_asyncio.

import json
import hashlib
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, render_template, request, send_from_directory, session
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


@app.errorhandler(413)
def payload_too_large(_exc):
    # Flask's default 413 page is HTML, which breaks frontend res.json() parsing.
    limit_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return (
        jsonify({"success": False, "error": f"Upload too large — max {limit_mb} MB per request."}),
        413,
    )

ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".html", ".md",
    ".csv", ".xlsx",
}
MAX_HISTORY = 3

Path(Config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(Config.CACHE_FOLDER).mkdir(parents=True, exist_ok=True)

router = QueryRouter()
_preindex_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preindex")


def _google_credential() -> str:
    """Read the Google ID token without colliding with HF's bearer token."""
    return (request.headers.get("X-Omnirag-Auth") or "").strip()


def _verify_google_credential(credential: str) -> dict:
    if not Config.GOOGLE_OAUTH_CLIENT_ID:
        raise RuntimeError("GOOGLE_OAUTH_CLIENT_ID is not configured")

    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2 import id_token

    claims = id_token.verify_oauth2_token(
        credential,
        GoogleRequest(),
        Config.GOOGLE_OAUTH_CLIENT_ID,
    )
    if not claims.get("sub") or claims.get("email_verified") is not True:
        raise ValueError("Google account is not verified")
    return claims


@app.before_request
def authenticate_request():
    """Require Google authentication for every account-data API route."""
    if request.method == "OPTIONS":
        return None
    if request.endpoint in {"static", "index", "login", "admin", "config_js", "healthz"}:
        return None
    if not Config.AUTH_REQUIRED:
        g.auth_user = {
            "sub": _client_session_id(),
            "email": "local@omnirag.test",
            "name": "Local user",
        }
        return None

    credential = _google_credential()
    if not credential:
        return jsonify({"error": "Authentication required."}), 401
    try:
        g.auth_user = _verify_google_credential(credential)
    except RuntimeError as exc:
        logger.error("[auth] %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.info("[auth] Google token rejected: %s", type(exc).__name__)
        return jsonify({"error": "Your Google session is invalid or expired."}), 401


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


def _account_key(claims: dict) -> str:
    """Return the non-reversible filesystem namespace for a verified user."""
    return hashlib.sha256(f"google:{claims['sub']}".encode()).hexdigest()[:32]


def _session_dir() -> Path:
    claims = getattr(g, "auth_user", None)
    if claims and claims.get("sub"):
        # Do not expose a raw Google identifier in the storage bucket path.
        account_key = _account_key(claims)
    else:
        account_key = _client_session_id()
    d = Path(Config.UPLOAD_FOLDER) / account_key
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
    # A request that explicitly names the selected document/report is already
    # self-contained.  Do not let conversation history replace its intent with
    # a previous calculation or fact lookup.
    if re.search(
        r"\b(?:summari[sz]e|summary|overview|describe)\b.*\b(?:document|report|pdf|file)\b",
        query_text,
        flags=re.IGNORECASE,
    ):
        return query_text
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
            reasoning_effort="none",
        )
        reformulated = resp.choices[0].message.content.strip()
        # Strip <think>...</think> block if present (e.g. from Qwen reasoning model)
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
    # Vercel serves its build-time config.js directly. Flask serves this
    # runtime version locally so a locally configured OAuth client ID reaches
    # the Google Identity Services script without being committed to source.
    api_base = "http://127.0.0.1:5000"
    script = (
        f"window.OMNIRAG_API_BASE_URL = {json.dumps(api_base)};\n"
        f"window.OMNIRAG_GOOGLE_CLIENT_ID = {json.dumps(Config.GOOGLE_OAUTH_CLIENT_ID)};\n"
    )
    return Response(script, mimetype="application/javascript")


def _is_admin(claims: dict | None = None) -> bool:
    claims = claims or getattr(g, "auth_user", {})
    email = str(claims.get("email") or "").strip().lower()
    return bool(email and email in Config.ADMIN_EMAILS)


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _is_admin():
            return jsonify({"error": "Administrator access is required."}), 403
        return view(*args, **kwargs)

    return wrapped


def _safe_account_directory(root: Path, account_key: str) -> Path:
    """Resolve a known account directory without allowing path traversal."""
    if not re.fullmatch(r"[a-f0-9]{32}", account_key):
        abort(404)
    root = root.resolve()
    target = (root / account_key).resolve()
    if target.parent != root:
        abort(404)
    return target


def _directory_size(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _admin_user_summary(profile: dict) -> dict:
    account_key = profile["account_key"]
    upload_dir = _safe_account_directory(Path(Config.UPLOAD_FOLDER), account_key)
    cache_root = Path(Config.CACHE_FOLDER) / "accounts"
    cache_dir = _safe_account_directory(cache_root, account_key)
    files = []
    if upload_dir.is_dir():
        for path in sorted(upload_dir.iterdir(), key=lambda item: item.name.lower()):
            try:
                if path.is_file() and allowed_file(path.name):
                    files.append({"name": path.name, "bytes": path.stat().st_size})
            except OSError:
                continue
    upload_bytes = sum(file["bytes"] for file in files)
    cache_bytes = _directory_size(cache_dir)
    return {
        **profile,
        "files": files,
        "upload_bytes": upload_bytes,
        "cache_bytes": cache_bytes,
        "total_bytes": upload_bytes + cache_bytes,
    }


def _delete_account_cache(account_key: str) -> None:
    cache_dir = _safe_account_directory(Path(Config.CACHE_FOLDER) / "accounts", account_key)
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)


def _delete_account_workspace(account_key: str) -> None:
    upload_dir = _safe_account_directory(Path(Config.UPLOAD_FOLDER), account_key)
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir)
    _delete_account_cache(account_key)


@app.route("/login")
def login():
    # Static page only — no auth flow wired up. On Vercel this file is served
    # directly as a static asset; this route exists so it's also reachable
    # from the local Flask dev server.
    return render_template("login.html")


@app.route("/admin")
def admin():
    # The static Vercel deployment serves admin.html directly. This route keeps
    # the owner console available from the local Flask development server too.
    return render_template("admin.html")


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


@app.route("/auth/me")
def auth_me():
    """Return the Google profile tied to the current private workspace."""
    claims = g.auth_user
    upload_dir = _session_dir()  # Provision account storage on first login.
    if Config.AUTH_REQUIRED:
        from admin_store import record_user

        record_user(_account_key(claims), claims)
    return jsonify(
        {
            "id": claims["sub"],
            "email": claims.get("email", ""),
            "name": claims.get("name") or claims.get("email", "OmniRAG user"),
            "picture": claims.get("picture", ""),
            "is_admin": _is_admin(claims),
        }
    )


@app.route("/files")
def files():
    """List documents belonging only to the authenticated Google account."""
    return jsonify({"files": _uploaded_files(_session_dir())})


@app.route("/admin/users")
@admin_required
def admin_users():
    from admin_store import users

    records = [_admin_user_summary(profile) for profile in users()]
    return jsonify(
        {
            "users": records,
            "totals": {
                "users": len(records),
                "upload_bytes": sum(record["upload_bytes"] for record in records),
                "cache_bytes": sum(record["cache_bytes"] for record in records),
                "total_bytes": sum(record["total_bytes"] for record in records),
            },
        }
    )


@app.route("/admin/users/<account_key>/files/<path:filename>", methods=["DELETE"])
@admin_required
def admin_delete_file(account_key: str, filename: str):
    upload_dir = _safe_account_directory(Path(Config.UPLOAD_FOLDER), account_key)
    safe_name = secure_filename(filename)
    if safe_name != filename or not allowed_file(safe_name):
        abort(404)
    file_path = upload_dir / safe_name
    if not file_path.is_file():
        abort(404)
    file_path.unlink()
    # Indexes and structured records derived from the deleted file must not be retained.
    _delete_account_cache(account_key)
    return jsonify({"success": True})


@app.route("/admin/users/<account_key>/workspace", methods=["DELETE"])
@admin_required
def admin_delete_workspace(account_key: str):
    _delete_account_workspace(account_key)
    return jsonify({"success": True})


@app.route("/admin/users/<account_key>", methods=["DELETE"])
@admin_required
def admin_forget_user(account_key: str):
    _delete_account_workspace(account_key)
    from admin_store import forget_user

    forget_user(account_key)
    return jsonify({"success": True})


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


def _process_saved_file(file_path: Path, errors: list, warnings: list, spreadsheet_status: dict):
    """Post-save ingestion for one already-written upload (spreadsheet + scan check)."""
    filename = file_path.name
    if file_path.suffix.lower() in {".csv", ".xlsx"}:
        try:
            from doc_loader import ingest_spreadsheet

            row_count = ingest_spreadsheet(file_path)
            spreadsheet_status[filename] = {
                "structured": "ready",
                "rows": row_count,
                "semantic": "indexing",
            }
        except Exception as exc:
            logger.exception("[upload] Spreadsheet ingestion failed for %s", filename)
            errors.append(f"{filename} — spreadsheet parsing failed: {exc}")
            spreadsheet_status[filename] = {"structured": "failed", "error": str(exc)}
    if _looks_like_scanned_pdf(file_path):
        warnings.append(f"{filename} looks image-based/scanned, so answers may take longer.")


def _finalize_upload(upload_dir: Path, uploaded: list, errors: list, warnings: list, spreadsheet_status: dict):
    """Shared response tail for /upload and /upload-chunk: history reset + preindex."""
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
        # One global queue prevents overlapping calls to a 1-RPS embedding API.
        _preindex_executor.submit(_preindex_basic_rag, text_files, upload_dir)

    return jsonify(
        {
            "success": True,
            "files": combined,
            "errors": errors,
            "warnings": warnings,
            "spreadsheets": spreadsheet_status,
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    upload_dir = _session_dir()
    uploaded = []
    errors = []
    warnings = []
    spreadsheet_status = {}

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
        _process_saved_file(file_path, errors, warnings, spreadsheet_status)

    return _finalize_upload(upload_dir, uploaded, errors, warnings, spreadsheet_status)


CHUNK_STAGING_DIR = ".chunks"
CHUNK_TTL_SECONDS = 3600  # sweep chunk dirs abandoned by a closed browser


def _sweep_stale_chunks(upload_dir: Path) -> None:
    """Remove chunk-staging dirs older than the TTL so aborted uploads can't leak disk."""
    staging_root = upload_dir / CHUNK_STAGING_DIR
    if not staging_root.is_dir():
        return
    cutoff = time.time() - CHUNK_TTL_SECONDS
    for d in staging_root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


@app.route("/upload-chunk", methods=["POST"])
def upload_chunk():
    """Receive one ≤4 MB slice of a larger file; reassemble on the final chunk.

    Lets uploads exceed the Vercel proxy's ~4.5 MB per-request body cap while
    the HF Space stays private (every chunk still flows through the token-
    injecting proxy). The reassembled file is capped at MAX_CONTENT_LENGTH.
    """
    upload_dir = _session_dir()

    raw_name = request.form.get("filename", "")
    try:
        chunk_index = int(request.form.get("chunk_index", ""))
        total_chunks = int(request.form.get("total_chunks", ""))
    except ValueError:
        return jsonify({"success": False, "error": "Invalid chunk metadata."}), 400

    # upload_id must be a UUID: this both validates the client and guarantees
    # the staging path can't traverse outside the chunk directory.
    try:
        upload_id = str(uuid.UUID(str(request.form.get("upload_id", ""))))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid upload id."}), 400

    if not raw_name or chunk_index < 0 or total_chunks < 1 or chunk_index >= total_chunks:
        return jsonify({"success": False, "error": "Invalid chunk metadata."}), 400

    filename = secure_filename(raw_name)
    if not filename or not allowed_file(filename):
        return jsonify({"success": False, "error": f"{raw_name} — unsupported file type"}), 400

    chunk = request.files.get("chunk")
    if not chunk:
        return jsonify({"success": False, "error": "Missing chunk data."}), 400

    if chunk_index == 0:
        _sweep_stale_chunks(upload_dir)

    staging = upload_dir / CHUNK_STAGING_DIR / upload_id
    staging.mkdir(parents=True, exist_ok=True)
    # Zero-pad so lexical sort == numeric order when reassembling.
    chunk.save(str(staging / f"{chunk_index:06d}"))

    # More chunks to come — acknowledge and wait.
    if chunk_index < total_chunks - 1:
        return jsonify({"success": True, "received": chunk_index, "pending": True})

    # Final chunk: verify completeness, enforce the size cap, then reassemble.
    parts = sorted(staging.glob("[0-9]" * 6))
    if len(parts) != total_chunks:
        shutil.rmtree(staging, ignore_errors=True)
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{filename} — upload incomplete "
                    f"({len(parts)}/{total_chunks} chunks received); please retry.",
                }
            ),
            400,
        )

    total_size = sum(p.stat().st_size for p in parts)
    if total_size > Config.MAX_CONTENT_LENGTH:
        shutil.rmtree(staging, ignore_errors=True)
        limit_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"{filename} — file too large "
                    f"({total_size // (1024 * 1024)} MB); max {limit_mb} MB.",
                }
            ),
            413,
        )

    file_path = upload_dir / filename
    with file_path.open("wb") as dest:
        for p in parts:
            with p.open("rb") as part:
                shutil.copyfileobj(part, dest)
    shutil.rmtree(staging, ignore_errors=True)

    errors: list = []
    warnings: list = []
    spreadsheet_status: dict = {}
    _process_saved_file(file_path, errors, warnings, spreadsheet_status)
    return _finalize_upload(upload_dir, [filename], errors, warnings, spreadsheet_status)


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

    # Compute an exact spreadsheet aggregate once at the boundary.  Agent
    # engines still answer the document portions of a mixed request, but this
    # verified result prevents a sampled vector row from being presented as a
    # workbook-wide maximum/minimum.
    from spreadsheet_query import try_structured_query
    verified_spreadsheet_result = try_structured_query(standalone_query, filenames, upload_dir)
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
            try:
                merged_answer = str(merge_llm.complete(merge_prompt)).strip()
            except Exception as exc:
                from utils import is_provider_failure
                if not is_provider_failure(exc):
                    raise
                from llama_index.llms.google_genai import GoogleGenAI
                logger.info("[merged] Mistral synthesis failed (%s); falling back to Gemini %s", type(exc).__name__, Config.GOOGLE_LLM)
                merged_answer = str(GoogleGenAI(
                    api_key=Config.GOOGLE_API_KEY,
                    model=Config.GOOGLE_LLM,
                    max_tokens=Config.MERGED_ANSWER_MAX_TOKENS,
                    max_retries=Config.GOOGLE_MAX_RETRIES,
                    is_function_calling_model=False,
                ).complete(merge_prompt)).strip()

            # Strip <think>...</think> block if present
            import re
            if "<think>" in merged_answer:
                merged_answer = re.sub(r"<think>.*?</think>", "", merged_answer, flags=re.DOTALL).strip()
                if "<think>" in merged_answer:
                    merged_answer = merged_answer.split("<think>")[0].strip()

            result = {
                "answer": merged_answer,
                "sources": text_result.get("sources", []),
                "thinking_steps": [],
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
    if verified_spreadsheet_result is not None:
        verified_answer = verified_spreadsheet_result["answer"]
        if verified_answer not in answer:
            answer = f"{verified_answer}\n\n{answer}".strip()
            existing_sources = result.get("sources", [])
            result["sources"] = verified_spreadsheet_result.get("sources", []) + existing_sources
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
