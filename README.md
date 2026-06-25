---
title: AI Video Assistant
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Transcribe, summarise and chat with any meeting video.
---

# 🎬 AI Video Assistant

🚀 **Live demo**: [ajayreddy0112-ai-video-assistant.hf.space](https://ajayreddy0112-ai-video-assistant.hf.space)

> Transcribe, summarise and chat with any meeting video — Whisper + LangChain RAG, wrapped in a Streamlit UI.

> ℹ️ **On the live demo, use the Upload tab.** YouTube blocks cloud-hosted IPs (HF Spaces, AWS, etc.) so URL mode is best used locally. Upload any short audio/video file to try the full pipeline.

Paste a YouTube URL or upload a short meeting video (≤ 20 min) and get:

- 📝 **Full transcript** (Whisper for English, Sarvam STT for Hinglish → English)
- 🏷️ A short, professional **meeting title**
- 📋 A bullet-point **executive summary**
- ✅ **Action items** with owners and deadlines
- 🔑 **Key decisions** captured
- ❓ **Open questions** flagged for follow-up
- 💬 A **history-aware RAG chat** to ask anything about the meeting

All exportable as `.txt`, `.md` and `.json`.

---

## 🧱 Architecture

```
┌─ Streamlit UI (app.py) ────────────────────────────────────────────────┐
│                                                                        │
│   Sidebar:  URL / Upload  +  language  +  duration guard               │
│      │                                                                 │
│      ▼                                                                 │
│   utils/audio_processor  ──►  yt-dlp │ ffmpeg → 10-min WAV chunks      │
│                                (master + chunks cleaned post-use)      │
│      │                                                                 │
│      ▼                                                                 │
│   core/transcriber       ──►  Whisper (en) or Sarvam STT-Translate (hi)│
│                                                                        │
│   core/summarizer        ──►  Mistral · map-reduce summary + title     │
│   core/extractor         ──►  Mistral · action items / decisions / Qs  │
│                                                                        │
│   core/vector_store      ──►  Chroma in-memory + BGE-small embeddings  │
│                                (UUID-per-build, MMR retrieval)         │
│   core/rag_engine        ──►  LCEL · history-aware rewrite → retrieve  │
│                                → ground → answer + sources             │
│                                (atexit + weakref auto-cleanup)         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Stack

| Layer        | Tech                                                   |
|--------------|--------------------------------------------------------|
| UI           | Streamlit (custom dark theme + CSS)                    |
| STT          | OpenAI Whisper (local) · Sarvam STT-Translate (API)    |
| LLM          | Mistral (`mistral-small-latest`) via `langchain-mistralai` |
| Embeddings   | `BAAI/bge-small-en-v1.5` (normalised)                  |
| Vector store | ChromaDB (in-memory, per-session)                      |
| Orchestration| LangChain LCEL                                         |
| Media        | `yt-dlp` · `pydub` · `ffmpeg`                          |
| Packaging    | `uv` + `pyproject.toml`                                |

---

## ✨ Design notes (why these choices)

- **MMR over pure similarity** — meetings paraphrase the same idea many times. MMR (`k=6`, `fetch_k=24`, `λ=0.5`) diversifies retrievals so the LLM gets broader coverage.
- **BGE-small-en-v1.5 over MiniLM** — ~5 MTEB points better at retrieval, only +40 MB, drop-in replacement.
- **History-aware query rewriting** — every follow-up question is rewritten into a standalone form (resolving pronouns like "his deadline" → "Bob's deadline") before retrieval. Skipped for the first message to keep latency down.
- **In-memory vector store** — single-meeting workflow, no cross-session library, ephemeral disk on free hosting tiers anyway. Eliminates the cross-video contamination bug that comes with a fixed `collection_name` on a persistent dir.
- **Three-layered cleanup** — explicit `cleanup()`, `weakref.finalize` (auto-fires on session GC), and `atexit` (process shutdown). Memory stays flat across rebuilds.
- **Source transparency** — every assistant answer shows a 📎 expander with the actual retrieved transcript chunks + the rewritten standalone question. No black box.
- **20-minute hard cap** — checked upfront (probes URL duration *before* downloading, checks file duration before conversion). No wasted bandwidth on rejected inputs.

---

## 🚀 Quickstart

### Prerequisites

- Python **3.11**
- [`ffmpeg`](https://ffmpeg.org/) on `PATH`
- A free [Mistral API key](https://console.mistral.ai/)

### Install

```bash
git clone <repo-url>
cd ai-video-assistant
cp .env.example .env          # then fill in MISTRAL_API_KEY

# preferred (fast, reproducible)
uv sync

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# UI
streamlit run app.py

# CLI
python main.py
```

---

## 🐳 Docker

```bash
docker build -t ai-video-assistant .
docker run --rm -p 8501:8501 --env-file .env ai-video-assistant
```

Then open <http://localhost:8501>.

---

## ☁️ Deploy

### HuggingFace Spaces (recommended)

This repo ships with the Space metadata header above and `packages.txt` declaring
the `ffmpeg` apt dependency. To deploy:

1. Create a new **Streamlit Space** on huggingface.co/new-space
2. Push this repo to it (Git remote)
3. In Space *Settings → Variables and secrets*, add:
   - `MISTRAL_API_KEY` (required)
   - `SARVAM_API_KEY` (optional, only for Hinglish input)

> Whisper `small` fits comfortably in the 16 GB free tier. For tighter envs,
> set `WHISPER_MODEL=base` or `tiny` (also via Space secrets).

### Streamlit Community Cloud

Works out of the box — `packages.txt` is picked up automatically. On the free
1 GB tier you'll want `WHISPER_MODEL=tiny`.

### Render / Railway / Fly.io

Use the included `Dockerfile`.

---

## ⚙️ Configuration

All settings live in `core/config.py` and read from the environment.

| Var                 | Default                  | Meaning                                  |
|---------------------|--------------------------|------------------------------------------|
| `MISTRAL_API_KEY`   | *(required)*             | Mistral chat LLM                         |
| `MISTRAL_MODEL`     | `mistral-small-latest`   | Override model name                      |
| `WHISPER_MODEL`     | `small`                  | `tiny` · `base` · `small` · `medium` …   |
| `SARVAM_API_KEY`    | *(only for Hinglish)*    | Sarvam STT API                           |
| `SARVAM_STT_MODEL`  | `saaras:v2.5`            | Sarvam model variant                     |
| `CHUNK_MINUTES`     | `10`                     | Audio chunk size in minutes              |
| `MAX_VIDEO_MINUTES` | `20`                     | Hard cap on input video duration         |
| `EMBEDDING_MODEL`   | `BAAI/bge-small-en-v1.5` | sentence-transformers model              |

---

## 📁 Project layout

```
ai-video-assistant/
├── app.py                  # Streamlit UI
├── main.py                 # CLI entry point
├── core/
│   ├── config.py           # Settings + .env loading
│   ├── llm.py              # Shared Mistral client (lru_cached)
│   ├── transcriber.py      # Whisper + Sarvam STT, chunk cleanup
│   ├── summarizer.py       # Map-reduce summary + title
│   ├── extractor.py        # Action items / decisions / questions
│   ├── vector_store.py     # In-memory Chroma + BGE embeddings + MMR
│   └── rag_engine.py       # History-aware LCEL chain + lifecycle
├── utils/
│   └── audio_processor.py  # yt-dlp + ffmpeg + duration guard + safe_remove
├── assets/
│   └── styles.css          # Streamlit dark theme
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── packages.txt            # apt deps for HF Spaces / Streamlit Cloud
├── pyproject.toml
└── requirements.txt
```

---

## 🧹 Resource lifecycle (what's cleaned up, when)

| Artefact                      | Cleaned up where                          | Triggered by                       |
|-------------------------------|-------------------------------------------|------------------------------------|
| Downloaded YouTube WAV        | `process_input` (try/finally)             | After chunking succeeds            |
| Audio chunks                  | `transcribe_all` (per-chunk + outer)      | After each chunk transcribed       |
| Streamlit temp upload         | `app.py` run block (finally)              | After pipeline ends, any outcome   |
| Chroma collection (in-memory) | `chain.cleanup()` + `weakref.finalize`    | New Analyse / Reset / session GC   |
| Chroma global system cache    | `SharedSystemClient.clear_system_cache()` | Same triggers as above             |
| Whisper / embedding models    | (kept in process for warm starts)         | Process exit                       |

The **🗑️ Reset session** button in the sidebar is the manual escape hatch.

---

## 📝 License

MIT
