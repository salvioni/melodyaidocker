"""Gemini Flash service for singer assignment in lyrics."""

import re

from google import genai

from melody.services.config import GEMINI_API_KEY

_MODEL = "gemini-3-flash-preview"
_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_enabled = False


async def assign_singers(lyrics: str, title: str, artist: str) -> list[dict[str, str]]:
    """Call Gemini Flash to assign singer tags to lyrics lines.

    Returns a list of ``{"line": ..., "singer": ...}`` dicts, one per non-empty
    line in the output.  If the API call fails the result is an empty list so
    that the pipeline can continue without singer data.
    """
    if not _enabled or _client is None:
        return []

    prompt = (
        "You are a karaoke metadata parser. I am going to give you the plain "
        f"lyrics to '{title}' by {artist}. Your job is to format these lyrics "
        "by adding bracketed singer tags (e.g. [Bruno Mars], [Anderson .Paak], "
        "[Both], [All]) above the lines they sing. Output ONLY the formatted "
        "lyrics, no other text.\n\n"
        f"{lyrics}"
    )

    try:
        response = await _client.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config={"temperature": 0.0},
        )
        return _parse_singer_tags(response.text or "")
    except Exception:
        # Non-critical — return empty so the pipeline isn't blocked.
        return []


_SINGER_TAG = re.compile(r"^\[([^\]]+)\]\s*$")


def _parse_singer_tags(text: str) -> list[dict[str, str]]:
    """Parse Gemini output into a flat list of ``{"line": ..., "singer": ...}``.

    The expected format is:

        [Singer A]
        Line one
        Line two
        [Singer B]
        Line three

    Each lyric line inherits the most recent singer tag.
    """
    result: list[dict[str, str]] = []
    current_singer: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SINGER_TAG.match(line)
        if m:
            current_singer = m.group(1)
        elif current_singer is not None:
            result.append({"line": line, "singer": current_singer})

    return result
