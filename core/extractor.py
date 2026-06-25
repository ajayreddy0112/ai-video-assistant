"""Pull action items, decisions and open questions out of a transcript."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm import get_llm


def _build_chain(system_prompt: str):
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{text}")]
        )
        | get_llm(temperature=0.2)
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    return _build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    ).invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    return _build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    ).invoke(transcript)


def extract_questions(transcript: str) -> str:
    return _build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    ).invoke(transcript)
