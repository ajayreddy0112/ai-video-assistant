"""Download/convert source media to WAV and slice into manageable chunks."""

from __future__ import annotations

import logging
import os

import yt_dlp
from pydub import AudioSegment

from core.config import settings

logger = logging.getLogger(__name__)


class VideoTooLongError(ValueError):
    """Raised when source media exceeds the configured maximum duration."""


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def safe_remove(path: str | None) -> None:
    """Best-effort delete — never raises."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning("Could not remove %s: %s", path, e)


def _enforce_duration(seconds: float, label: str) -> None:
    max_seconds = settings.max_video_minutes * 60
    if seconds > max_seconds:
        raise VideoTooLongError(
            f"{label} is {seconds / 60:.1f} min — limit is "
            f"{settings.max_video_minutes} min. Please trim it and retry."
        )


_YDL_COMMON_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "retries": 5,
    "fragment_retries": 5,
    "socket_timeout": 30,
    "nocheckcertificate": True,
    "geo_bypass": True,
    "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    },
}


def download_youtube_audio(url: str) -> str:
    with yt_dlp.YoutubeDL({**_YDL_COMMON_OPTS, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration")

    if duration is None:
        raise ValueError("Could not determine video duration. Please try a different URL.")
    _enforce_duration(float(duration), "Video")

    output_template = os.path.join(str(settings.download_dir), "%(title)s.%(ext)s")
    ydl_opts = {
        **_YDL_COMMON_OPTS,
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        for ext in (".webm", ".m4a", ".mp3", ".opus"):
            filename = filename.replace(ext, ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)
    _enforce_duration(audio.duration_seconds, "File")
    audio = audio.set_channels(1).set_frame_rate(16000)
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = None) -> list[str]:
    audio = AudioSegment.from_wav(wav_path)
    minutes = chunk_minutes or settings.chunk_minutes
    chunk_ms = minutes * 60 * 1000

    chunks: list[str] = []
    base = os.path.splitext(wav_path)[0]
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = f"{base}_chunk_{i}.wav"
        audio[start : start + chunk_ms].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> list[str]:
    if _is_url(source):
        logger.info("Detected URL — downloading audio")
        wav_path = download_youtube_audio(source)
    else:
        logger.info("Detected local file — converting to WAV")
        wav_path = convert_to_wav(source)

    try:
        logger.info("Chunking audio")
        chunks = chunk_audio(wav_path)
        logger.info("Audio ready — %d chunk(s)", len(chunks))
        return chunks
    finally:
        safe_remove(wav_path)
