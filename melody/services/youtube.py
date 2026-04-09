"""YouTube audio download using yt-dlp's Python API."""

import asyncio
import os
import re
from pathlib import Path

import yt_dlp

from melody.services.config import TEMP_DIR

# Limit concurrent downloads to avoid YouTube rate-limiting.
_download_semaphore = asyncio.Semaphore(2)

_YDL_BASE_OPTS: dict = {
    # Target ~128 kbps — good balance for voice/music, keeps file size small for performance
    # "format": "bestaudio[abr<=128]/bestaudio",
    # Prefer Android/iOS clients — fewer restrictions than web client
    "extractor_args": {"youtube": {"player_client": ["android", "ios"]}},
    # Pass browser cookies to appear as a logged-in session
    # "cookiesfrombrowser": ("chrome",),
    # Retry / backoff
    "retries": 5,
    "fragment_retries": 5,
    "retry_sleep_functions": {"http": lambda n: min(2**n, 30)},
    # Convert to mp3 after download
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "0",
        }
    ],
    "quiet": True,
    "no_warnings": True,
}


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url
    return None


async def download_audio(url: str) -> tuple[str, float | None]:
    """Download audio from a YouTube URL.

    Returns ``(file_path, duration_seconds)``.
    Caches by video ID — skips download if the mp3 already exists.
    """
    video_id = _extract_video_id(url)
    if video_id is None:
        raise ValueError(f"Could not extract YouTube video ID from: {url}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    output_path = os.path.join(TEMP_DIR, video_id)
    mp3_path = output_path + ".mp3"

    if os.path.isfile(mp3_path):
        duration = await _get_duration(url)
        return mp3_path, duration

    async with _download_semaphore:
        # Run blocking yt-dlp in a thread pool so we don't block the event loop
        await asyncio.to_thread(_run_download, url, output_path)

    duration = await _get_duration(url)
    return mp3_path, duration


def _run_download(url: str, output_path: str) -> None:
    opts = {
        **_YDL_BASE_OPTS,
        "outtmpl": output_path + ".%(ext)s",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


async def _get_duration(url: str) -> float | None:
    try:
        info = await asyncio.to_thread(_extract_info, url)
        return info.get("duration")
    except Exception:
        return None


def _extract_info(url: str) -> dict:
    opts = {**_YDL_BASE_OPTS, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False) or {}
