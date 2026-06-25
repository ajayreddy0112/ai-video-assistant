"""Map-reduce summarisation and title generation over a transcript."""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from core.llm import get_llm

logger = logging.getLogger(__name__)


def _split(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.summary_chunk_size,
        chunk_overlap=settings.summary_chunk_overlap,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm(temperature=0.3)

    map_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Summarise this portion of a meeting transcript concisely."),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    chunks = _split(transcript)
    logger.info("Summarising %d chunk(s)", len(chunks))
    partials = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined_text = "\n\n".join(partials)

    reduce_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert meeting summariser. Combine these partial summaries "
                    "into one polished, professional meeting summary as concise bullet points.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return reduce_chain.invoke(combined_text)


def generate_title(transcript: str) -> str:
    llm = get_llm(temperature=0.3)
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional "
                    "meeting title (max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    return chain.invoke(transcript[:2000]).strip().strip('"')
