"""Catalog routes — song browsing, searching, and lyrics retrieval."""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from melody.models.database import SongsRow
from melody.services import music_provider as music_provider_svc
from melody.services import lrclib
from melody.services import supabase_client as db
from melody.routes.types import (
    HomeResponse,
    LyricsData,
    LyricsResponse,
    ProcessedSong,
    SongSearchResult,
)

router = APIRouter()


def _build_fallback_song_result(song: SongsRow) -> SongSearchResult:
    provider_id = song.get("provider_id") or ""
    provider = song.get("provider", "user_upload")

    youtube_url = (
        f"https://www.youtube.com/watch?v={provider_id}"
        if provider == "youtube" and provider_id
        else ""
    )
    youtube_thumb = (
        f"https://i.ytimg.com/vi/{provider_id}/hqdefault.jpg"
        if provider == "youtube" and provider_id
        else ""
    )

    return SongSearchResult(
        providerId=provider_id or None,
        provider=provider,
        title=song.get("title", ""),
        artist=song.get("artist", ""),
        artistId="",
        artistThumbnail=youtube_thumb,
        url=youtube_url,
        art=youtube_thumb,
        thumbnail=youtube_thumb,
        releaseDate="",
        processedSong=ProcessedSong.model_validate(song),
    )


# ---------------------------------------------------------------------------
# GET /home
# ---------------------------------------------------------------------------


@router.get("/home", tags=["home"])
async def home() -> HomeResponse:
    """Return all songs that have been processed (status=done)."""
    all_songs = await db.get_all_songs()
    done = [s for s in all_songs if s.get("status") == "done"]

    async def enrich(song: SongsRow) -> SongSearchResult:
        if song.get("provider") == "youtube" and song.get("provider_id"):
            try:
                provider_id = song["provider_id"]
                if provider_id is None:
                    raise ValueError("Missing provider_id")
                song_data, _ = await music_provider_svc.get_song(provider_id)
                return SongSearchResult.model_validate(
                    music_provider_svc.song_to_search_dict(
                        song_data, processed_song=song
                    )
                )
            except Exception:
                print(
                    f"Error enriching song {song['id']} from provider data:",
                )
                pass
        return _build_fallback_song_result(song)

    songs = await asyncio.gather(*[enrich(s) for s in done])
    return HomeResponse(songs=list(songs))


# ---------------------------------------------------------------------------
# GET /artist/{artist_id}/songs
# ---------------------------------------------------------------------------


@router.get("/artist/{artist_id}/songs", tags=["artist"])
async def get_artist_songs(
    artist_id: str,
    per_page: int = Query(default=20, ge=1, le=50),
    page: int = Query(default=1, ge=1),
) -> list[SongSearchResult]:
    songs = await music_provider_svc.get_artist_songs(
        artist_id, per_page=per_page, page=page
    )
    provider_ids = [str(s.get("id", "")) for s in songs if s.get("id")]
    processed = await db.get_songs_by_provider_ids("youtube", provider_ids)
    processed_map = {r["provider_id"]: r for r in processed}

    return [
        SongSearchResult.model_validate(
            music_provider_svc.song_to_search_dict(
                song, processed_song=processed_map.get(str(song.get("id", "")))
            )
        )
        for song in songs
    ]


# ---------------------------------------------------------------------------
# GET /search/{query}
# ---------------------------------------------------------------------------


@router.get("/search/{query}", tags=["search"])
async def search_songs(query: str) -> list[SongSearchResult]:
    provider_results = await music_provider_svc.search_songs(query)
    provider_ids = [str(s.get("id", "")) for s in provider_results if s.get("id")]
    processed = await db.get_songs_by_provider_ids("youtube", provider_ids)
    processed_map = {r["provider_id"]: r for r in processed}

    return [
        SongSearchResult.model_validate(
            music_provider_svc.song_to_search_dict(
                song, processed_song=processed_map.get(str(song.get("id", "")))
            )
        )
        for song in provider_results
    ]


# ---------------------------------------------------------------------------
# GET /lyrics/{song_id}  and  GET /lyrics/provider/{provider_id}
# ---------------------------------------------------------------------------


@router.get("/lyrics/{song_id}", tags=["lyrics"])
async def get_lyrics_by_id(song_id: int) -> LyricsResponse:
    """Return synchronized lyrics for a processed song."""
    try:
        karaoke = await db.download_karaoke_json(song_id)
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"Karaoke data not found for song {song_id}."
        )

    if not isinstance(karaoke, dict):
        raise HTTPException(
            status_code=500, detail=f"Invalid karaoke payload for song {song_id}."
        )

    lyrics = karaoke.get("lyrics", {})
    return LyricsResponse(
        source="database",
        lyrics=LyricsData.model_validate(lyrics),
    )


@router.get("/lyrics/provider/{provider_id}", tags=["lyrics"])
async def get_lyrics_by_provider_id(provider_id: str) -> LyricsResponse:
    """Return lyrics from LRCLIB for a provider song (no word-level timing)."""
    song_data, _ = await music_provider_svc.get_song(provider_id)
    artist = (song_data.get("primary_artist") or {}).get("name", "")
    title = song_data.get("title") or ""

    lyrics_text = await lrclib.fetch_lyrics(artist_name=artist, track_name=title)
    if lyrics_text is None:
        raise HTTPException(status_code=404, detail="Lyrics not found on LRCLIB.")

    return LyricsResponse(
        source="lrclib",
        lyrics=LyricsData(text=lyrics_text, source="human"),
    )
