"""Supabase database and storage client."""

import asyncio
import json
from pathlib import Path
from typing import cast

import httpx
from supabase import Client, create_client

from melody.models.database import Json, SongProvider, SongsInsert, SongsRow
from melody.services.config import (
    SIGNED_URL_EXPIRY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
)

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


# ---------------------------------------------------------------------------
# Songs table
# ---------------------------------------------------------------------------


async def upsert_song(
    *,
    song_id: int | None = None,
    provider: SongProvider,
    provider_id: str | None,
    title: str,
    artist: str,
    status: str = "processing",
    owner_id: str | None = None,
) -> SongsRow:
    row: SongsInsert = {
        "provider": provider,
        "title": title,
        "artist": artist,
        "status": status,
    }
    if song_id is not None:
        row["id"] = song_id
    if provider_id is not None:
        row["provider_id"] = provider_id
    if owner_id is not None:
        row["owner_id"] = owner_id

    result = await asyncio.to_thread(
        lambda: (
            get_supabase().from_("songs").upsert(cast(dict[str, Json], row)).execute()
        )
    )
    rows = result.data
    return cast(SongsRow, rows[0])


async def get_song_by_id(song_id: int) -> SongsRow | None:
    result = await asyncio.to_thread(
        lambda: (
            get_supabase().from_("songs").select().eq("id", song_id).limit(1).execute()
        )
    )
    rows = result.data
    return cast(SongsRow | None, rows[0] if rows else None)


async def get_song_by_provider(
    provider: SongProvider, provider_id: str
) -> SongsRow | None:
    result = await asyncio.to_thread(
        lambda: (
            get_supabase()
            .from_("songs")
            .select()
            .eq("provider", provider)
            .eq("provider_id", provider_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    )
    rows = result.data
    return cast(SongsRow | None, rows[0] if rows else None)


async def update_song_status(song_id: int, status: str) -> SongsRow:
    result = await asyncio.to_thread(
        lambda: (
            get_supabase()
            .from_("songs")
            .update({"status": status})
            .eq("id", song_id)
            .execute()
        )
    )
    data = result.data
    if isinstance(data, list):
        return cast(SongsRow, data[0])
    return cast(SongsRow, data)


async def get_all_songs() -> list[SongsRow]:
    result = await asyncio.to_thread(
        lambda: get_supabase().from_("songs").select().execute()
    )
    return cast(list[SongsRow], result.data)


async def get_songs_by_provider_ids(
    provider: SongProvider, provider_ids: list[str]
) -> list[SongsRow]:
    result = await asyncio.to_thread(
        lambda: (
            get_supabase()
            .from_("songs")
            .select()
            .eq("provider", provider)
            .in_("provider_id", provider_ids)
            .execute()
        )
    )
    return cast(list[SongsRow], result.data)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


async def upload_file(storage_path: str, file_path: str) -> None:
    """Upload a file to Supabase Storage with retry logic."""
    data = Path(file_path).read_bytes()
    content_type = _guess_content_type(file_path)

    # Retry with exponential backoff for connection errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await asyncio.to_thread(
                lambda: (
                    get_supabase()
                    .storage.from_("songs")
                    .upload(
                        storage_path,
                        data,
                        file_options={"content-type": content_type, "upsert": "true"},
                    )
                )
            )
            return  # Success
        except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError):
            if attempt == max_retries - 1:
                raise  # Last attempt failed
            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2**attempt
            await asyncio.sleep(wait_time)


async def download_file(storage_path: str) -> bytes:
    result = await asyncio.to_thread(
        lambda: get_supabase().storage.from_("songs").download(storage_path)
    )
    return result


async def create_signed_urls(paths: list[str]) -> list[str]:
    result = await asyncio.to_thread(
        lambda: (
            get_supabase()
            .storage.from_("songs")
            .create_signed_urls(paths, SIGNED_URL_EXPIRY)
        )
    )
    return [r["signedURL"] for r in result]


async def download_karaoke_json(song_id: int) -> Json:
    raw = await download_file(f"{song_id}/karaoke.json")
    return cast(Json, json.loads(raw.decode("utf-8")))


def _guess_content_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")
