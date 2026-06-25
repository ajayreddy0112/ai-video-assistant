"""Retrieval-augmented Q&A over a single meeting transcript."""

from __future__ import annotations

import atexit
import logging
import weakref
from typing import Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm
from core.vector_store import build_vector_store, get_retriever

logger = logging.getLogger(__name__)


def _release_store(store) -> None:
    """Drop a Chroma collection and purge Chroma's global system cache."""
    try:
        store.delete_collection()
    except Exception as e:
        logger.warning("Could not drop collection: %s", e)
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception as e:
        logger.warning("Could not clear Chroma system cache: %s", e)


_ANSWER_PROMPT = """You are a precise meeting analyst. Answer the user's question using ONLY the meeting context below.

Rules:
- If the answer is not clearly supported by the context, reply exactly:
  "I could not find this information in the meeting transcript."
- Be concise. Prefer short paragraphs or bullet points.
- Quote speakers verbatim only when it strengthens the answer (e.g. 'As X said: "…"').
- Do NOT invent names, numbers, dates, or decisions.
- If the context conflicts with itself, briefly note the conflict.
- Do NOT refer to "chunks", "excerpts", "the transcript", "the context", "as mentioned above",
  or any meta-references. Write as if you naturally know the meeting — never reference
  where the information came from. Answer the question directly.

Meeting context:
{context}"""


_REWRITE_PROMPT = """Given the conversation so far and the user's latest question, rewrite the question so it can stand alone (resolving pronouns like "he", "it", "that decision" against the chat history).

If the latest question is already self-contained, return it unchanged. Output ONLY the rewritten question — no preamble.

Conversation:
{history}

Latest question: {question}

Standalone question:"""


class RagAnswer(TypedDict):
    answer: str
    sources: list[Document]
    standalone_question: str


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(empty)"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    )


def build_rag_chain(transcript: str) -> Callable[..., RagAnswer]:
    store = build_vector_store(transcript)
    retriever = get_retriever(store)

    answer_chain = (
        ChatPromptTemplate.from_messages(
            [("system", _ANSWER_PROMPT), ("human", "{question}")]
        )
        | get_llm(temperature=0.1)
        | StrOutputParser()
    )

    rewrite_chain = (
        ChatPromptTemplate.from_template(_REWRITE_PROMPT)
        | get_llm(temperature=0.0)
        | StrOutputParser()
    )

    def invoke(question: str, history: list[dict] | None = None) -> RagAnswer:
        history = history or []
        standalone = question
        if history:
            standalone = rewrite_chain.invoke(
                {"history": _format_history(history), "question": question}
            ).strip().strip('"')
            if standalone.lower().startswith("standalone question:"):
                standalone = standalone.split(":", 1)[1].strip()

        docs = retriever.invoke(standalone)
        answer = answer_chain.invoke(
            {"context": _format_docs(docs), "question": standalone}
        )
        return {"answer": answer, "sources": docs, "standalone_question": standalone}

    def cleanup() -> None:
        """Release the vector store. Idempotent — safe to call repeatedly."""
        finalizer()  # weakref.finalize() detaches after first call

    # Safety net 1: when the chain becomes unreachable (e.g. Streamlit GCs an idle
    # session), free the store automatically.
    finalizer = weakref.finalize(invoke, _release_store, store)

    # Safety net 2: free on process shutdown (Ctrl-C, container stop).
    atexit.register(finalizer)

    invoke.cleanup = cleanup  # type: ignore[attr-defined]
    return invoke


def ask_question(
    rag_chain: Callable[..., RagAnswer],
    question: str,
    history: list[dict] | None = None,
) -> RagAnswer:
    logger.info("Q: %s", question)
    result = rag_chain(question, history)
    preview = result["answer"][:120] + ("…" if len(result["answer"]) > 120 else "")
    logger.info("A: %s", preview)
    return result
