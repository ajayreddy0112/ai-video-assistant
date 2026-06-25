"""In-memory Chroma + sentence-transformers vector store for RAG retrieval.

We keep the store in-memory so each pipeline run gets a fresh, isolated index.
The store is held alive by the LCEL chain returned from rag_engine.build_rag_chain,
which lives in st.session_state for the duration of the user's session.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(transcript: str) -> Chroma:
    logger.info("Building in-memory vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(splitter.split_text(transcript))
    ]
    logger.info("Indexed %d chunk(s)", len(docs))

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"meeting_{uuid.uuid4().hex[:8]}",
    )


def get_retriever(store: Chroma, k: int | None = None):
    """MMR retriever — returns diverse, relevant chunks instead of near-duplicates."""
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or settings.rag_top_k,
            "fetch_k": settings.rag_fetch_k,
            "lambda_mult": settings.rag_mmr_lambda,
        },
    )
