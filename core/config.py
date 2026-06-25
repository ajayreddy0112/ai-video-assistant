"""Centralised runtime configuration. Loads `.env` once and exposes typed settings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "downloads"

DOWNLOAD_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # ── API keys ────────────────────────────────────────────────────────────
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")

    # ── Models ──────────────────────────────────────────────────────────────
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    sarvam_model: str = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ── Endpoints ───────────────────────────────────────────────────────────
    sarvam_url: str = "https://api.sarvam.ai/speech-to-text-translate"

    # ── Pipeline tuning ─────────────────────────────────────────────────────
    chunk_minutes: int = int(os.getenv("CHUNK_MINUTES", "10"))
    max_video_minutes: int = int(os.getenv("MAX_VIDEO_MINUTES", "20"))
    sarvam_piece_seconds: int = 25
    summary_chunk_size: int = 3000
    summary_chunk_overlap: int = 200
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_top_k: int = 6
    rag_fetch_k: int = 24
    rag_mmr_lambda: float = 0.5

    # ── Storage ─────────────────────────────────────────────────────────────
    download_dir: Path = DOWNLOAD_DIR


settings = Settings()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s · %(name)s · %(levelname)s · %(message)s",
        datefmt="%H:%M:%S",
    )
