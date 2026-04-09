"""LRCLIB open-source lyrics API client."""

import re

import httpx

# https://lrclib.net/docs
_BASE = "https://lrclib.net/api"


async def fetch_lyrics(
    *,
    artist_name: str,
    track_name: str,
    album_name: str | None = None,
    duration_seconds: float | None = None,
) -> str | None:
    """Fetch plain or synced lyrics from LRCLIB.

    Returns the best available lyrics string, or ``None`` if not found.
    Prefers synced lyrics (LRC format) over plain text.
    """
    params: dict = {"artist_name": artist_name, "track_name": track_name}
    if album_name:
        params["album_name"] = album_name
    if duration_seconds is not None:
        params["duration"] = int(duration_seconds)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_BASE}/get", params=params)

    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    data = resp.json()
    # synced = data.get("syncedLyrics")
    plain = data.get("plainLyrics")

    # if synced:
    #     return synced
    return plain


async def fetch_synced_lyrics(
    *, artist_name: str, track_name: str
) -> list[dict] | None:
    """Fetch synced lyrics and parse them into a list of ``{time, text}`` dicts.

    Returns ``None`` if no synced lyrics are available.
    Each dict has ``startSeconds`` (float) and ``text`` (str).
    """
    lyrics = await fetch_lyrics(artist_name=artist_name, track_name=track_name)
    if not lyrics or not lyrics.strip().startswith("["):
        return None
    return parse_lrc(lyrics)


def parse_lrc(lrc: str) -> list[dict]:
    """Parse an LRC string into a list of ``{startSeconds, text}`` dicts."""
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    lines = []
    for line in lrc.splitlines():
        m = pattern.match(line.strip())
        if m:
            minutes, seconds, text = int(m.group(1)), float(m.group(2)), m.group(3)
            lines.append({"startSeconds": minutes * 60 + seconds, "text": text.strip()})
    return lines
