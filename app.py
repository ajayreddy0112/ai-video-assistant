"""Streamlit UI — AI Video Assistant. Transcribe, summarise & chat with any meeting."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from core.config import settings, setup_logging
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import VideoTooLongError, process_input, safe_remove

setup_logging()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styles ─────────────────────────────────────────────────────────────────────
_CSS_FILE = Path(__file__).parent / "assets" / "styles.css"
st.markdown(f"<style>{_CSS_FILE.read_text()}</style>", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
_DEFAULT_STATE = {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "trigger_sample": False,
}
for k, v in _DEFAULT_STATE.items():
    st.session_state.setdefault(k, v)

# ─── Constants ──────────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("audio", "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title", "🏷️", "Title Generation"),
    ("summary", "📋", "Summarisation"),
    ("extract", "🔍", "Extraction"),
    ("rag", "🧠", "RAG Indexing"),
]

SAMPLE_VIDEO_PATH = Path(__file__).parent / "samples" / "fake_meeting.mp4"
HAS_SAMPLE = SAMPLE_VIDEO_PATH.exists()


# ─── Helpers ────────────────────────────────────────────────────────────────────
def _dot_class(state: str) -> str:
    return {"active": "dot-active", "done": "dot-done", "error": "dot-error"}.get(
        state, "dot-pending"
    )


def _render_status(container) -> None:
    rows = "".join(
        f"""<div class="status-bar">
                <div class="status-dot {_dot_class(st.session_state.pipeline_steps.get(key, "pending"))}"></div>
                <span>{icon}&nbsp;&nbsp;{label}</span>
            </div>"""
        for key, icon, label in PIPELINE_STEPS
    )
    container.markdown(rows, unsafe_allow_html=True)


def _save_upload(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    return tmp.name


def _results_payload(r: dict) -> str:
    return json.dumps(
        {k: v for k, v in r.items() if k != "rag_chain"}, indent=2, ensure_ascii=False
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Source</span>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input mode",
        ["Upload", "URL"],
        horizontal=True,
        label_visibility="collapsed",
    )

    url_value = ""
    uploaded_file = None
    if input_mode == "URL":
        url_value = st.text_input(
            "YouTube URL or path",
            placeholder="https://youtube.com/watch?v=…",
            label_visibility="collapsed",
        )
        st.caption(
            "⚠️ YouTube blocks cloud-hosted IPs (HF Spaces, AWS, etc.). "
            "URL mode may fail here — works reliably **locally**. "
            "**Upload mode** is recommended for the live demo."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video",
            type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "ogg"],
            label_visibility="collapsed",
        )

    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    st.caption(f"⏱️ Max video length: {settings.max_video_minutes} min")
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.result:
        if st.button("🗑️ Reset session", use_container_width=True, type="secondary"):
            prev = st.session_state.result.get("rag_chain")
            if prev and hasattr(prev, "cleanup"):
                prev.cleanup()
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<span class="badge badge-cyan">Pipeline Status</span>', unsafe_allow_html=True
    )
    status_slot = st.empty()
    _render_status(status_slot)


# ─── Main: Hero ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─── Pipeline ───────────────────────────────────────────────────────────────────
def _resolve_source() -> str | None:
    if input_mode == "URL":
        return url_value.strip() or None
    if uploaded_file is None:
        return None
    return _save_upload(uploaded_file)


def _set_step(key: str, state: str) -> None:
    st.session_state.pipeline_steps[key] = state
    _render_status(status_slot)


trigger_run = run_btn or st.session_state.trigger_sample

if trigger_run:
    is_sample = st.session_state.trigger_sample
    st.session_state.trigger_sample = False

    if is_sample:
        source = str(SAMPLE_VIDEO_PATH)
    else:
        source = _resolve_source()

    if not source:
        st.error("Please provide a URL or upload a file.")
    else:
        upload_temp = source if (input_mode == "Upload" and not is_sample) else None

        if st.session_state.result and st.session_state.result.get("rag_chain"):
            prev = st.session_state.result["rag_chain"]
            if hasattr(prev, "cleanup"):
                prev.cleanup()

        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        _render_status(status_slot)

        progress_slot = st.empty()
        progress_slot.info("⚙️ Pipeline running — see sidebar for live status…")

        try:
            _set_step("audio", "active")
            chunks = process_input(source)
            _set_step("audio", "done")

            _set_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            _set_step("transcript", "done")

            _set_step("title", "active")
            title = generate_title(transcript)
            _set_step("title", "done")

            _set_step("summary", "active")
            summary = summarize(transcript)
            _set_step("summary", "done")

            _set_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            _set_step("extract", "done")

            _set_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            _set_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            progress_slot.success("✅ Analysis complete!")

        except VideoTooLongError as e:
            for k, _, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "error"
            _render_status(status_slot)
            progress_slot.error(f"⏱️ {e}")

        except Exception as e:
            for k, _, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "error"
            _render_status(status_slot)
            progress_slot.error(f"❌ {type(e).__name__}: {e}")

        finally:
            safe_remove(upload_temp)


# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # ── Title + downloads ───────────────────────────────────────────────────
    head_left, head_right = st.columns([3, 1], gap="medium")
    with head_left:
        st.markdown(
            f"""
        <div class="card">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
                {r["title"]}
            </div>
        </div>""",
            unsafe_allow_html=True,
        )
    with head_right:
        st.markdown(
            '<div class="card"><div class="card-title">⬇️ Exports</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Transcript .txt",
            r["transcript"],
            file_name=f"{r['title'][:40]}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Summary .md",
            f"# {r['title']}\n\n## Summary\n{r['summary']}\n\n"
            f"## Action Items\n{r['action_items']}\n\n"
            f"## Key Decisions\n{r['key_decisions']}\n\n"
            f"## Open Questions\n{r['open_questions']}\n",
            file_name=f"{r['title'][:40]}_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "All .json",
            _results_payload(r),
            file_name=f"{r['title'][:40]}_full.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tabbed results ──────────────────────────────────────────────────────
    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
        ["📋 Summary", "✅ Actions", "🔑 Decisions", "❓ Questions", "📝 Transcript"]
    )

    with tab_summary:
        st.markdown(
            f'<div class="card"><div class="card-title">📋 Meeting Summary</div>'
            f'<div class="card-content">{r["summary"]}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_actions:
        st.markdown(
            f'<div class="card"><div class="card-title">✅ Action Items</div>'
            f'<div class="card-content">{r["action_items"]}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_decisions:
        st.markdown(
            f'<div class="card"><div class="card-title">🔑 Key Decisions</div>'
            f'<div class="card-content">{r["key_decisions"]}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_questions:
        st.markdown(
            f'<div class="card"><div class="card-title">❓ Open Questions</div>'
            f'<div class="card-content">{r["open_questions"]}</div></div>',
            unsafe_allow_html=True,
        )
    with tab_transcript:
        st.markdown(
            f'<div class="transcript-box">{r["transcript"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── RAG chat ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;'
        'font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        chat_html = ['<div class="chat-container">']
        for msg in st.session_state.chat_history:
            is_user = msg["role"] == "user"
            label_cls = "user-label" if is_user else "bot-label"
            bubble_cls = "user-bubble" if is_user else "bot-bubble"
            align = "flex-end" if is_user else "flex-start"
            label = "You" if is_user else "🤖 Assistant"
            chat_html.append(
                f'<div class="chat-msg" style="align-items:{align}">'
                f'<span class="chat-label {label_cls}">{label}</span>'
                f'<div class="chat-bubble {bubble_cls}">{msg["content"]}</div>'
            )
            sources = msg.get("sources") or []
            if not is_user and sources:
                rewritten = msg.get("standalone_question", "")
                meta = (
                    f'<div class="sources-meta">Resolved: <em>{rewritten}</em></div>'
                    if rewritten
                    else ""
                )
                snippets = "".join(
                    f'<div class="source-chunk"><span class="source-tag">#{i + 1}</span> {s}</div>'
                    for i, s in enumerate(sources)
                )
                chat_html.append(
                    f'<details class="sources-details">'
                    f'<summary>📎 {len(sources)} source chunk(s)</summary>'
                    f"{meta}{snippets}"
                    f"</details>"
                )
            chat_html.append("</div>")
        chat_html.append("</div>")
        st.markdown("".join(chat_html), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:2rem">'
            '<div style="font-size:2rem;margin-bottom:0.5rem">💬</div>'
            '<div style="color:var(--text-muted);font-size:0.85rem">'
            "Ask anything about your meeting transcript</div></div>",
            unsafe_allow_html=True,
        )

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1], gap="small")
        with col_input:
            user_input = st.text_input(
                "Your question",
                placeholder="What were the main decisions made?",
                label_visibility="collapsed",
            )
        with col_send:
            send_btn = st.form_submit_button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            out = ask_question(
                r["rag_chain"],
                user_input.strip(),
                st.session_state.chat_history,
            )
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input.strip()}
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": out["answer"],
                "sources": [d.page_content for d in out["sources"]],
                "standalone_question": out["standalone_question"],
            }
        )
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # ── Empty state ─────────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem 1.5rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:420px;line-height:1.7">
            Upload an audio/video file in the sidebar (or paste a YouTube URL),
            choose your language, and hit <strong>Analyse</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Whisper / Sarvam</span>
            <span class="badge badge-cyan">LangChain · Mistral</span>
            <span class="badge badge-green">Chroma RAG</span>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    if HAS_SAMPLE:
        st.markdown(
            """
        <div style="display:flex;align-items:center;justify-content:center;gap:1rem;margin:1.5rem 0 0.75rem;color:var(--text-muted);font-size:0.7rem;letter-spacing:0.2em">
            <div style="flex:1;max-width:160px;height:1px;background:var(--border)"></div>
            <span>OR TRY THE DEMO</span>
            <div style="flex:1;max-width:160px;height:1px;background:var(--border)"></div>
        </div>""",
            unsafe_allow_html=True,
        )
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            if st.button(
                "🎬  Try a sample meeting (1 min, no upload needed)",
                use_container_width=True,
                key="sample_btn",
            ):
                st.session_state.trigger_sample = True
                st.rerun()
        st.markdown(
            '<div style="text-align:center;color:var(--text-muted);font-size:0.72rem;'
            'margin-top:0.5rem">A bundled 1-minute mock product sync with named'
            ' attendees, action items, decisions and open questions.</div>',
            unsafe_allow_html=True,
        )
