---
title: OmniRAG Multi-Engine Explorer
emoji: "🔎"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# OmniRAG

OmniRAG is a Flask and LlamaIndex application for asking questions about uploaded documents and images. It automatically selects a suitable RAG/agent workflow, supports follow-up questions, and renders Markdown and LaTeX answers in the browser.

Supported uploads: PDF, TXT, Markdown, HTML, CSV, XLSX, PNG, JPG, JPEG, and GIF.

## What it does

- Routes text questions to Basic RAG, Router, Sub-Question, Multi-Document, or ReAct workflows.
- Analyses image-only uploads with a vision model.
- Combines image and document answers when both are uploaded.
- Uses a shared per-file BGE-M3 vector index: an unchanged file is embedded once and reused across all text engines.
- Keeps separate summary indexes for overview questions; summary indexes do not create a second set of embeddings.
- Handles scanned PDFs with an OCR fallback.
- Renders valid LaTeX output with KaTeX.

## Architecture

```text
Browser
  │
  ├─ local: Flask serves the UI at http://127.0.0.1:5000
  └─ deployment: Vercel frontend → Hugging Face Space API
                                      │
                                      ├─ BGE-M3 shared vector indexes
                                      ├─ persisted summary indexes
                                      └─ Groq, Gemini, and Mistral APIs
```

The frontend sends a browser-generated `X-Omnirag-Session-Id` header. The backend uses it to keep uploads and chat history isolated by browser session.

## Engines

| Engine | Use case | Answer model / flow |
| --- | --- | --- |
| Basic RAG | Narrow factual lookup | Mistral over hybrid vector + BM25 retrieval |
| Router Engine | Overview or a mix of overview and exact facts | Groq selects summary or vector retrieval; Gemini fallback on Groq rate limits |
| Sub-Question | Multi-part or cross-document questions | Groq decomposes into document-level questions; Gemini fallback |
| Multi-Document Agent | Explicit multi-document mode | Per-document agents queried in parallel, then coordinated by Gemini |
| Multi-Modal | Image-only questions | Groq vision model |
| ReAct Agent | Thinking mode | Gemini tool-using agent with document search tools |
| Merged | Both images and text uploaded | Runs image and text analysis, then synthesizes a combined answer |

All text engines use the same `mistral-embed` API model and shared vector-cache format. LLMs may differ by engine; that does not affect vector-index compatibility.

## Indexing and caching

On upload, the backend starts a background Basic RAG pre-index job. It processes text files sequentially so a 1-RPS Mistral limit is not exceeded by concurrent index jobs.

```text
Upload PDF
  → build shared_vector cache once
  → any engine loads the same vector index
  → Router / Multi-Document may additionally build a summary index
```

If a query reaches an engine while that file’s vector index is still building, the engine waits on the shared cache lock, then loads the finished index. It does not embed the file a second time.

CSV/XLSX files use a dual ingestion path. Exact rows, headers, and formulas are stored immediately in `CACHE_FOLDER/spreadsheets.sqlite3`; exact lookups and aggregations can run there without waiting for embeddings. In parallel, only sheet profiles, formula groups, and narrative cells enter the semantic index. Ordinary numeric rows are not duplicated into the vector store.

Cache keys include file content, chunk size, and embedding model. Changing any of them creates a new cache automatically.

The embedding client is initialized while Flask/Gunicorn starts, before the first upload.
The default Mistral indexing profile uses 512-token chunks, 64-token overlap,
and batches of 32 to reduce API request count. These can be adjusted with
`CHUNK_SIZE`, `CHUNK_OVERLAP`, and `EMBED_BATCH_SIZE`. On Hugging Face, uploaded
files and vector indexes use `/data/uploads` and `/data/cache`; attaching persistent
Space storage allows unchanged PDFs to reuse their indexes across restarts.

## Quick start

Prerequisites: Python 3.12+ and API keys for Mistral, Google, and Groq.

```bash
pip install -r requirements.txt
copy .env.example .env
python app.py
```

On macOS/Linux, replace `copy` with `cp`.

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

For scanned PDFs, install Tesseract locally and ensure it is available on `PATH`. The Docker image installs it automatically.

## Configuration

Copy `.env.example` to `.env` and set at least:

```env
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
FLASK_SECRET_KEY=...
```

Useful optional settings:

```env
# Storage
UPLOAD_FOLDER=uploads
CACHE_FOLDER=cache

# Embeddings and retrieval
EMBED_MODEL=BAAI/bge-m3

# Models
MISTRAL_LLM=mistral-large-latest
GOOGLE_LLM=gemini-3.1-flash-lite-preview
REACT_PRIMARY_LLM=gemini-2.5-flash
GROQ_LLM=qwen/qwen3.6-27b
GROQ_VISION_LLM=qwen/qwen3.6-27b
GROQ_ROUTER_LLM=llama-3.3-70b-versatile
GROQ_SUBQUESTION_LLM=llama-3.3-70b-versatile

# Cross-origin frontend deployment
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
SESSION_COOKIE_SAMESITE=None
SESSION_COOKIE_SECURE=true
```

`GOOGLE_API_KEY2` is optional and is used only as the ReAct engine’s secondary-key fallback. `COHERE_API_KEY` and the legacy Gemma variables are optional.

## API

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Application UI |
| `/healthz` | GET | Health/status response |
| `/api-status` | GET | Configured-provider status |
| `/upload` | POST | Upload one or more files for the current session |
| `/remove-file` | POST | Remove one uploaded file |
| `/clear-files` | POST | Remove all session uploads |
| `/new-chat` | POST | Clear chat history and keep uploads |
| `/query` | POST | Send a question and receive an answer |

`/query` accepts JSON with `query` and optional boolean `multi_doc` and `thinking` fields.

## Deploying the frontend and backend

### Hugging Face Space backend

Use a Docker Space. The included [Dockerfile](Dockerfile) runs Gunicorn on port `7860`.

Set secrets for the API keys and `FLASK_SECRET_KEY`. Set variables similar to:

```env
PORT=7860
UPLOAD_FOLDER=/data/uploads
CACHE_FOLDER=/data/cache
CORS_ORIGINS=https://your-vercel-app.vercel.app
SESSION_COOKIE_SAMESITE=None
SESSION_COOKIE_SECURE=true
```

Free Spaces can sleep and their default filesystem is ephemeral. Use persistent storage if vector caches and uploads must survive a restart.

### Vercel frontend

Deploy [`frontend/`](frontend) as the Vercel root directory. The included
`frontend/api/backend/[...path].js` Vercel Function is a same-origin proxy for
a private Hugging Face Space. Set these Vercel environment variables:

```env
HF_SPACE_URL=https://your-space.hf.space
HF_SPACE_READ_TOKEN=hf_...
```

`HF_SPACE_READ_TOKEN` must be a fine-grained Hugging Face read token scoped to
the Space. It is used only by the Vercel Function and is never sent to the
browser. The frontend build routes production requests to `/api/backend`; do
not set `OMNIRAG_API_BASE_URL` to the private `.hf.space` URL.

## Repository layout

```text
app.py                    Flask routes, upload/session handling, merged flow
config.py                 Environment-driven configuration
router.py                 Rule-based and Groq-assisted engine routing
answer_format.py          Shared Markdown/LaTeX answer instructions
shared_vector_index.py    Per-file shared vector cache and build lock
retrieval.py              Hybrid vector + BM25 retriever and answer template
doc_loader.py             File loading and OCR fallback
index_cache.py            Cache keys, metadata, and locks
model_cache.py            In-process Hugging Face embedding-model cache
utils.py                  Retry, file classification, and source helpers
engines/
  basic_rag.py            Basic RAG
  router_engine.py        Summary/vector router engine
  subquestion.py          Question decomposition engine
  multi_document.py       Parallel document-agent workflow
  multimodal.py           Vision analysis
  react_agent.py          Tool-using ReAct workflow
frontend/                 Static Vercel frontend
demo-images/              README screenshots
cache/                    Local generated indexes (ignored by Git)
uploads/                  Local session uploads (ignored by Git)
```

## Troubleshooting

**First answer is slow** — the BGE-M3 model may need to load and new files must be embedded once. Later requests reuse the in-process model and shared vector indexes.

**A PDF is slow or has missing text** — it may be scanned; OCR has to run before retrieval.

**Formula does not render** — answers must contain valid `$...$`, `$$...$$`, `\(...\)`, or `\[...\]` delimiters. OmniRAG asks every engine for LaTeX; malformed source text can still require a follow-up request.

**Vercel cannot call the API** — check `HF_SPACE_URL` and `HF_SPACE_READ_TOKEN` in Vercel, then inspect the `/api/backend/api-status` function response. Do not configure a browser-visible direct private HF URL.

## License

MIT.
