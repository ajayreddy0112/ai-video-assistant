"""Single source of truth for the Mistral chat client used across the app."""

from __future__ import annotations

from functools import lru_cache

from langchain_mistralai import ChatMistralAI

from core.config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.3) -> ChatMistralAI:
    if not settings.mistral_api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Add it to your .env file or environment."
        )
    return ChatMistralAI(
        model=settings.mistral_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=temperature,
    )
