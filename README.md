# 🦙 LlamaIndex Multi-Engine Explorer

A production-ready Flask application that showcases **6 distinct LlamaIndex query engines** behind a single unified interface. Upload documents, ask questions, and watch the smart router automatically pick the best engine — or manually toggle Multi-Document and ReAct Thinking modes.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/flask-3.0%2B-green?logo=flask)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12%2B-purple)

---

## ✨ Features

- **6 Query Engines** — Basic RAG, Router Engine, Sub-Question, Multi-Document Agent, Multi-Modal, and ReAct Agent
- **Smart Routing** — Rule-based + LLM classification (Groq/Llama-4-Scout) automatically picks the best engine for each query
- **Multi-Provider Architecture** — Orchestrates Mistral, Google Gemma/Gemini, Groq, and Cohere APIs
- **Robust Document Loading** — Multi-layered pipeline (SimpleDirectoryReader -> PyMuPDF -> pypdf -> **Local OCR**)
- **Local OCR Support** — Automatic Tesseract OCR fallback for scanned or image-only PDFs
- **AgentWorkflow Pattern** — Uses LlamaIndex 0.12+ `FunctionAgent` for coordinated multi-document reasoning
- **Dual-LLM Multi-Doc Engine** — Orchestrated by Google/Gemma, synthesized by Groq/Llama-4-Scout for speed
- **Conversation Memory** — Follow-up questions are reformulated into standalone queries using Groq
- **Index Caching** — Built indexes are persisted to disk, with smart invalidation for PDFs to prevent empty results
- **Robust Retry Logic** — Exponential backoff on API rate limits, transient 500/502/503 errors, and malformed responses
- **Dark Mode UI** — Modern Bootstrap 5 dark theme with collapsible thinking steps and source citations

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Flask App (app.py)                    │
│                                                          │
│  User Query ──► Smart Router (router.py) ──► Engine      │
│                     │                          │         │
│            Rule-based checks              Engine Result  │
│            + Groq LLM fallback            ──► Response   │
└──────────────────────────────────────────────────────────┘

Smart Router Decision Flow:
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Thinking   │ YES │  ReAct Agent │     │  Multi-Doc    │
│  Mode on?   │────►│  (Gemini)    │     │  Agent Mode?  │
└──────┬──────┘     └──────────────┘     └──────┬────────┘
       │ NO                                     │ YES
       ▼                                        ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Images     │ YES │  Multimodal  │     │  Multi-Doc    │
│  only?      │────►│  (Groq)      │     │  AgentWorkflow│
└──────┬──────┘     └──────────────┘     │  (Gemma/Groq) │
       │ NO                              └───────────────┘
       ▼
┌─────────────┐     ┌──────────────┐
│  Images +   │ YES │   Merged     │
│  Text?      │────►│  (MM + Text) │
└──────┬──────┘     └──────────────┘
       │ NO
       ▼
┌─────────────────────────────────────┐
│  Groq LLM classifies query type:   │
│  • basic_rag      (factual lookup)  │
│  • subquestion    (comparison)      │
│  • router_engine  (summary/overview)│
└─────────────────────────────────────┘
```

### Multi-Document Agent Architecture (Dual-LLM)
The Multi-Document engine uses a tiered orchestration strategy:
1. **Top-Level Agent** (`Google Gemma 4-31B`): Orchestrates across all per-document agents.
2. **Per-Doc Agents** (`Google Gemma 4-31B`): Each document has a `FunctionAgent` with vector and summary tools.
3. **Synthesis** (`Groq Llama 4 Scout`): Fast final answering from retrieved chunks (with Gemma fallback if rate-limited).
```

---

## 🔧 Engines

| Engine | LLM Provider | Model | Use Case |
|---|---|---|---|
| **Basic RAG** | Mistral | `mistral-large-latest` | Simple factual lookups from a single document |
| **Router Engine** | Google | `gemma-4-31b-it` | Intelligently routes between vector search and summary index |
| **Sub-Question** | Google | `gemma-4-31b-it` | Decomposes complex queries across multiple documents |
| **Multi-Doc Agent** | Google + Groq | `Gemma` + `Llama-4-Scout` | **AgentWorkflow**: Per-file agents coordinated by a top-level agent |
| **Multi-Modal** | Groq | `llama-4-scout-17b-16e` | Vision-based image analysis |
| **ReAct Agent** | Google | `gemini-2.5-flash` | Step-by-step tool reasoning with visible thought process |

### Supporting Services

| Service | Provider | Purpose |
|---|---|---|
| **Query Routing** | Groq / Llama-4-Scout | Classifies ambiguous queries into engine labels |
| **Embeddings (RAG)** | HuggingFace | `BAAI/bge-base-en-v1.5` for Basic RAG and ReAct |
| **Embeddings (Multi-Doc)** | Mistral | `mistral-embed` for Multi-Document and Sub-Question |
| **Follow-up Reformulation** | Groq | Rewrites follow-up questions as standalone queries |
| **Reranking** | Cohere *(optional)* | Multi-document result reranking |

---

## 🚀 Quick Start

### 1. Clone & Navigate

```bash
cd third_party/LlamaIndex/app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key          # For Gemini 2.5 Flash (ReAct agent)
GOOGLE_API_KEY_GEMMA=your_gemma_api_key     # For Gemma 4-31B (Multi-Doc, Router, SubQuestion)
GROQ_API_KEY=your_groq_api_key

# Optional
COHERE_API_KEY=your_cohere_api_key          # For reranking
```

> **Where to get API keys:**
> - **Mistral** → [console.mistral.ai](https://console.mistral.ai/)
> - **Google** → [aistudio.google.com](https://aistudio.google.com/apikey)
> - **Groq** → [console.groq.com](https://console.groq.com/)
> - **Cohere** → [dashboard.cohere.com](https://dashboard.cohere.com/)

### 4. Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📁 Project Structure

```
app/
├── app.py                  # Flask application & routes
├── config.py               # Centralized configuration & env vars
├── doc_loader.py           # Robust loading pipeline with Tesseract OCR fallback
├── router.py               # Smart query router (rule-based + LLM)
├── utils.py                # Retry decorators, file classifiers, helpers
├── index_cache.py          # Index persistence & cache management (PDF invalidation)
├── model_cache.py          # Singleton HuggingFace embedding cache
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
│
├── engines/
│   ├── __init__.py         # Engine registry & get_engine()
│   ├── basic_rag.py        # Mistral + HuggingFace embeddings
│   ├── router_engine.py    # Vector + Summary dual-index with LLM selector
│   ├── subquestion.py      # Query decomposition across documents
│   ├── multi_document.py   # Per-document agent architecture
│   ├── multimodal.py       # Groq vision for image analysis
│   └── react_agent.py      # ReAct reasoning with tool use
│
├── templates/
│   └── index.html          # Single-page app template
│
├── static/
│   ├── css/style.css       # Dark theme styles
│   └── js/main.js          # Frontend logic (upload, chat, toggles)
│
├── uploads/                # Per-session uploaded files (auto-created)
└── cache/                  # Persisted vector/summary indexes (auto-created)
```

---

## 🎯 Usage Guide

### Basic Flow

1. **Upload documents** — Drag & drop or click "Browse" in the sidebar
2. **Ask a question** — Type in the input bar and press Enter
3. **View results** — See the chosen engine, router reasoning, answer, thinking steps, and sources

### Engine Modes

| Mode | How to Activate | Best For |
|---|---|---|
| **Auto (default)** | Just ask a question | The router picks the best engine |
| **Multi-Document** | Toggle in sidebar | Cross-document comparison with per-file agents |
| **Thinking (ReAct)** | Toggle in sidebar | Step-by-step reasoning with visible tool calls |

### Supported File Types

| Category | Extensions |
|---|---|
| **Text** | `.pdf`, `.txt`, `.html`, `.md` |
| **Image** | `.png`, `.jpg`, `.jpeg`, `.gif` |

> **Tip:** Upload images + text together — the router automatically merges multimodal and text analysis.

---

## ⚙️ Configuration

All configuration is centralized in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1024` | Token chunk size for document splitting |
| `SIMILARITY_TOP_K` | `8` | Number of similar nodes to retrieve |
| `GOOGLE_LLM` | `gemma-4-31b-it` | LLM for Multi-Doc, Router, Sub-Question |
| `GEMINI_LLM` | `gemini-2.5-flash` | LLM for ReAct Agent |
| `MISTRAL_LLM` | `mistral-large-latest` | LLM for Basic RAG |
| `GROQ_LLM` | `llama-4-scout-17b-16e-instruct` | LLM for routing & multimodal |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Mistral embedding model |
| `MAX_CONTENT_LENGTH` | `50 MB` | Maximum upload file size |

---

## 🔄 Retry & Error Handling

The application handles Google API instability (common with Gemma 4-31B on free tier) through **layered retry**:

| Layer | Scope | Retries | Backoff |
|---|---|---|---|
| **GoogleGenAI `max_retries`** | Per individual API call | 5 | Built-in tenacity |
| **`_invoke_agent()`** | Per agent workflow invocation | 5 | 2s → 4s → 8s → 16s → 16s |
| **`@with_retry` decorator** | Per engine `run()` call | 3 | Exponential (3s–20s) |

**Retryable errors:** `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable`, `MALFORMED_RESPONSE`, `terminated early`

---

## 🐍 Python Compatibility

| Python Version | Status | Notes |
|---|---|---|
| **3.12** | ✅ Fully supported | Recommended |
| **3.13** | ✅ Fully supported | — |
| **3.14** | ✅ Supported | `nest_asyncio` removed; async runs in dedicated threads via `concurrent.futures + asyncio.run()` |

> **Note:** Python 3.14 changed `asyncio` internals such that `nest_asyncio` can no longer patch the event loop. The application uses a thread-based async execution pattern to remain compatible.

---

## 📓 Companion Notebooks

Each engine in this app is based on a standalone Jupyter notebook in the parent directory:

| Notebook | Engine |
|---|---|
| `Basic_RAG_With_LlamaIndex.ipynb` | Basic RAG |
| `Router_Query_Engine.ipynb` | Router Engine |
| `SubQuestion_Query_Engine.ipynb` | Sub-Question |
| `Multi_Document_Agents.ipynb` | Multi-Document Agent |
| `Multi_Modal.ipynb` | Multi-Modal |
| `ReAct_Agent.ipynb` | ReAct Agent |

---

## 🛠️ Troubleshooting

### Google API 500 errors

The Gemma 4-31B model on Google's free tier frequently returns transient 500 errors, especially with large PDFs. Solutions:

- **Wait and retry** — The built-in retry logic handles most transient failures
- **Use a separate API key** — Set `GOOGLE_API_KEY_GEMMA` with a fresh key dedicated to Gemma
- **Switch to Gemini 2.5 Flash** — Change `GOOGLE_LLM` in `config.py` to `gemini-2.5-flash` for better stability
- **Use smaller documents** — TXT/MD files produce less context than large PDFs

### Scanned PDFs / OCR Issues

If you upload a scanned PDF and see "Empty PDF text", ensure you have Tesseract OCR installed:
- **Windows**: `winget install UB-Mannheim.TesseractOCR`
- **Linux**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`
- Make sure the `tessdata` path matches what's in `doc_loader.py` or set `TESSDATA_PREFIX`.

### Groq Rate Limits (429)

The application uses Groq for fast answering. If you hit rate limits, the Multi-Document engine will automatically fall back to Google/Gemma. For other engines, the built-in retry logic will wait for the rate limit window to reset (typically 60-120 seconds).

### Missing API key warnings

The app shows a yellow banner on startup listing any missing keys. At minimum, you need `MISTRAL_API_KEY`, `GOOGLE_API_KEY`, and `GROQ_API_KEY`.

### Slow first query

The first query downloads the HuggingFace embedding model (~420 MB) and builds the vector index. Subsequent queries use the cached model and index.

---

## 📄 License

This project is part of the [Claude Cookbooks](https://github.com/anthropics/claude-cookbooks) collection.
