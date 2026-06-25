# CLAUDE.md

Guide for Claude Code working in this repo. Read this before making non-trivial changes.

## What this project is

A Streamlit app that ingests a meeting video (YouTube URL or upload, capped at 20 min), transcribes it, summarises it, extracts action items / decisions / open questions, and lets the user chat with the transcript via a history-aware RAG pipeline.

Target audience: a single demo user on free hosting (HF Spaces). Not multi-tenant. Not a long-running service.

## Run / dev commands

| Task              | Command                                   |
|-------------------|-------------------------------------------|
| Install deps      | `uv sync`                                 |
| Run UI            | `.venv/bin/streamlit run app.py`          |
| Run CLI           | `.venv/bin/python main.py`                |
| Build Docker      | `docker build -t ai-video-assistant .`    |
| Health check URL  | `http://localhost:8501/_stcore/health`    |

There are no tests, no linter config, no CI. Don't add them unless asked.

## Architecture (one paragraph)

`app.py` → `utils.audio_processor.process_input` (downloads/converts/chunks audio) → `core.transcriber.transcribe_all` (Whisper or Sarvam) → `core.{summarizer,extractor}` (Mistral via shared `core.llm.get_llm`) → `core.rag_engine.build_rag_chain` (returns an LCEL callable with `.cleanup` attribute) → results held in `st.session_state["result"]`. The CLI mirror is `main.py`. All config flows through `core.config.settings` (a frozen dataclass) and `.env` is loaded exactly once at `core/config.py` import.

## Conventions to follow

- **One source of truth for LLM clients**: import `get_llm` from `core.llm`. Do **not** instantiate `ChatMistralAI` anywhere else.
- **Config goes in `core/config.py`**: any magic number or env var lives on the `Settings` dataclass. Don't add `os.getenv` calls scattered through modules.
- **Logging, not print**: every module has `logger = logging.getLogger(__name__)`. Use it. `setup_logging()` is called once from CLI / Streamlit entry.
- **Type hints on public functions**. Keep `from __future__ import annotations` at the top of new files.
- **Best-effort cleanup uses `safe_remove`** from `utils/audio_processor.py` — never raises.

## Critical design decisions — don't undo these without asking

1. **In-memory vector store** (`core/vector_store.py`: `Chroma.from_documents(...)` has **no** `persist_directory`). The earlier persistent version had a cross-video contamination bug because `Chroma.from_documents` appends to existing collections when a fixed `collection_name` is reused. If a future task wants a "past meetings library", that's a feature add, not a fix — flag it explicitly.

2. **UUID-per-build collection name** (`meeting_<uuid>`). Even in-memory, Chroma's `SharedSystemClient` is process-global. Unique names guarantee isolation between concurrent builds in the same process.

3. **Three-layer cleanup in `core/rag_engine.py`**:
   - `chain.cleanup()` — explicit, idempotent
   - `weakref.finalize(invoke, _release_store, store)` — auto-fires when the chain is GC'd
   - `atexit.register(finalizer)` — process shutdown
   `_release_store` must call **both** `store.delete_collection()` AND `SharedSystemClient.clear_system_cache()`. `delete_collection` alone leaks ~7-8 MB per rebuild because Chroma's shared system cache keeps segment state. Verified with `psutil` (see commit history).

4. **20-minute hard cap** is enforced **upfront** (`utils/audio_processor.py`: probes YouTube duration with `yt_dlp` *without* downloading; checks `AudioSegment.duration_seconds` before converting local files). Don't move this check to after download / conversion — it defeats the bandwidth-saving purpose.

5. **History-aware rewrite is skipped on the first turn** (`core/rag_engine.py: invoke()`). Don't always rewrite — the extra LLM call would double first-question latency for no benefit.

6. **RAG context format is plain transcript snippets separated by `---`**. **Do not** re-add `[Chunk N]` tags. We tried that; the LLM imitated them as citations ("As mentioned in Chunk 5") in its answers. Source transparency lives in the UI's 📎 expander, not in the LLM's output.

7. **Embedding model: `BAAI/bge-small-en-v1.5` with `normalize_embeddings=True`**. BGE requires normalisation for sensible cosine scores. If you swap models, check whether the new one needs prefix instructions ("Represent this sentence for retrieval:") or normalisation.

8. **CSS lives in `assets/styles.css`**, loaded by `app.py` via `st.markdown(<style>...)`. Don't inline new CSS in `app.py`. The existing `--accent`, `--accent-2`, `--surface`, etc. CSS variables are the design system — reuse them.

## Deploy

Primary target: **HuggingFace Spaces** (the YAML header in `README.md` configures it).
Secondary: any container host via `Dockerfile`.

Free-tier disk is ephemeral on every restart — that's *fine* for this app because nothing is supposed to persist.

## Gotchas you'll hit

- **basedpyright / cSpell warnings in the IDE are noise** — basedpyright isn't pointing at `.venv`, and cSpell flags every ML package name. Don't try to "fix" them by changing code; they're environmental.
- **Streamlit's session lifecycle has no `on_session_end` hook**. That's why we rely on `weakref.finalize` for the "user closed the tab" case.
- **`uv sync` after manual venv corruption**: if `openai-whisper` was pip-installed first with distutils, `uv` can't uninstall it. Fix is `rm -rf .venv && uv sync`.
- **Don't commit the `.env`**. `.env.example` is the template; the real `.env` is gitignored.

## Common asks → where to make the change

| Ask                                  | File(s) to touch                                    |
|--------------------------------------|-----------------------------------------------------|
| Change duration cap                  | `core/config.py` (`max_video_minutes`)              |
| Change embedding model               | `core/config.py` (`embedding_model`) — flushes cache; check normalisation |
| Change RAG retrieval params          | `core/config.py` (`rag_*`)                          |
| Tweak any prompt                     | `core/{summarizer,extractor,rag_engine}.py`         |
| Add a new export format              | `app.py` results section (download buttons)         |
| Visual theme                         | `assets/styles.css` + `.streamlit/config.toml`      |
| Add an env var                       | `core/config.py` Settings + `.env.example` + README |

## Things to NOT add unless explicitly asked

- Test suite, ruff config, pre-commit, Makefile — out of scope for this portfolio project.
- Multi-meeting library / persistent vector store — feature, not bugfix; ask first.
- Auth, multi-user namespacing — single-tenant by design.
- Background workers / Celery / queues — Streamlit runs synchronously and that's fine for ≤20-min videos.
- Comments explaining *what* code does. Comments only for non-obvious *why* (existing in `core/rag_engine.py` `_release_store` is a good example).
