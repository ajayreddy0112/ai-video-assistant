"""Speech-to-text. Whisper for English, Sarvam STT-Translate for Hinglish."""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests
import whisper
from pydub import AudioSegment

from core.config import settings
from utils.audio_processor import safe_remove

logger = logging.getLogger(__name__)

_whisper_model: Optional["whisper.Whisper"] = None


def _load_whisper() -> "whisper.Whisper":
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", settings.whisper_model)
        _whisper_model = whisper.load_model(settings.whisper_model)
        logger.info("Whisper model loaded.")
    return _whisper_model


def _transcribe_whisper(chunk_path: str) -> str:
    model = _load_whisper()
    result = model.transcribe(chunk_path, task="transcribe")
    return str(result["text"]).strip()


def _send_to_sarvam(piece_path: str) -> str:
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not set. Required for Hinglish transcription.")

    headers = {"api-subscription-key": settings.sarvam_api_key}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": settings.sarvam_model, "with_diarization": "false"}
        response = requests.post(
            settings.sarvam_url,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        logger.error("Sarvam returned %s: %s", response.status_code, response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


def _transcribe_sarvam(chunk_path: str) -> str:
    """Sarvam sync API caps at 30s, so we slice into shorter pieces and concatenate."""
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = settings.sarvam_piece_seconds * 1000
    total = (len(audio) + piece_ms - 1) // piece_ms

    parts: list[str] = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece_path = f"{chunk_path}_sv_{i}.wav"
        audio[start : start + piece_ms].export(piece_path, format="wav")
        try:
            logger.info("Sarvam piece %d/%d", i + 1, total)
            parts.append(_send_to_sarvam(piece_path))
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(parts).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return _transcribe_sarvam(chunk_path)
    return _transcribe_whisper(chunk_path)


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    logger.info("Transcribing %d chunk(s) with %s", len(chunks), engine)

    parts: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            logger.info("Chunk %d/%d", i + 1, len(chunks))
            try:
                parts.append(transcribe_chunk(chunk, language=language))
            finally:
                safe_remove(chunk)
    finally:
        for chunk in chunks:
            safe_remove(chunk)

    logger.info("Transcription complete.")
    return " ".join(parts).strip()
