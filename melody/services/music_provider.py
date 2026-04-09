"""Music metadata/search provider backed by ytmusicapi."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, TypedDict

from melody.models.database import SongsRow
from ytmusicapi import YTMusic


class MediaEntry(TypedDict):
    provider: str
    url: str


class PrimaryArtist(TypedDict):
    id: str
    name: str
    image_url: str
    header_image_url: str


class AlbumInfo(TypedDict):
    id: str
    name: str


class ProviderSong(TypedDict):
    id: str
    title: str
    url: str
    song_art_image_url: str
    song_art_image_thumbnail_url: str
    release_date_for_display: str
    album: AlbumInfo
    primary_artist: PrimaryArtist
    media: list[MediaEntry]


class SongSearchPayload(TypedDict):
    providerId: str | None
    provider: str
    title: str
    artist: str
    artistId: str
    artistThumbnail: str
    url: str
    art: str
    thumbnail: str
    releaseDate: str
    processedSong: SongsRow | None


@lru_cache(maxsize=1)
def _get_client() -> YTMusic:
    # Public, unauthenticated client.
    return YTMusic()


def _first_thumbnail(item: dict[str, Any]) -> str:
    thumbs = item.get("thumbnails")
    if not isinstance(thumbs, list) or not thumbs:
        return ""
    last = thumbs[-1]
    if not isinstance(last, dict):
        return ""
    url = last.get("url")
    return str(url) if isinstance(url, str) else ""


def _normalize_song(item: dict[str, Any]) -> ProviderSong:
    video_id = str(item.get("videoId") or "")
    artists = item.get("artists") if isinstance(item.get("artists"), list) else []
    first_artist = artists[0] if artists else {}
    artist = first_artist if isinstance(first_artist, dict) else {}
    album_raw = item.get("album") if isinstance(item.get("album"), dict) else {}
    year = item.get("year")
    url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

    # Construct YouTube thumbnail URLs directly from video ID
    art = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg" if video_id else ""
    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""

    # Try to extract artist thumbnail from ytmusicapi response
    artist_thumb = _first_thumbnail(item) or _first_thumbnail(artist) or thumbnail

    album: AlbumInfo = {
        "id": str(album_raw.get("id") or ""),
        "name": str(album_raw.get("name") or ""),
    }
    primary_artist: PrimaryArtist = {
        "id": str(artist.get("id") or ""),
        "name": str(artist.get("name") or ""),
        "image_url": artist_thumb,
        "header_image_url": artist_thumb,
    }
    media: list[MediaEntry] = [{"provider": "youtube", "url": url}] if url else []

    return {
        "id": video_id,
        "title": str(item.get("title") or ""),
        "url": url,
        "song_art_image_url": art,
        "song_art_image_thumbnail_url": thumbnail,
        "release_date_for_display": str(year) if year else "",
        "album": album,
        "primary_artist": primary_artist,
        "media": media,
    }


async def search_songs(query: str) -> list[ProviderSong]:
    """Search YouTube Music for songs matching *query*."""

    def _search() -> list[ProviderSong]:
        client = _get_client()
        results = client.search(query, filter="songs", limit=20)
        return [
            _normalize_song(r)
            for r in results
            if isinstance(r, dict) and r.get("videoId")
        ]

    return await asyncio.to_thread(_search)


async def get_song(song_id: str) -> tuple[ProviderSong, list[MediaEntry]]:
    """Fetch a song by YouTube video ID.

    Returns ``(song_dict, media_list)`` where *media_list* contains provider
    objects like ``{"provider": "youtube", "url": "https://..."}``
    """

    def _fetch() -> ProviderSong:
        client = _get_client()
        # Use a single-song watch playlist to obtain rich metadata around a
        # concrete video id.
        playlist = client.get_watch_playlist(videoId=song_id, limit=1)
        tracks = playlist.get("tracks") if isinstance(playlist, dict) else []
        if not isinstance(tracks, list) or not tracks:
            raise ValueError(f"Song not found for video id: {song_id}")
        first = tracks[0]
        if not isinstance(first, dict):
            raise ValueError(f"Invalid song payload for video id: {song_id}")
        return _normalize_song(first)

    song = await asyncio.to_thread(_fetch)
    return song, song.get("media", [])


async def get_artist_songs(
    artist_id: str, *, per_page: int = 20, page: int = 1
) -> list[ProviderSong]:
    """Fetch songs for an artist by YouTube Music browse ID."""

    def _fetch() -> list[ProviderSong]:
        client = _get_client()
        artist = client.get_artist(artist_id)
        songs_obj = artist.get("songs") if isinstance(artist, dict) else {}
        songs_dict = songs_obj if isinstance(songs_obj, dict) else {}
        results = (
            songs_dict.get("results")
            if isinstance(songs_dict.get("results"), list)
            else []
        )

        normalized = [
            _normalize_song(r)
            for r in results
            if isinstance(r, dict) and r.get("videoId")
        ]

        start = max(0, (page - 1) * per_page)
        end = start + per_page
        return normalized[start:end]

    return await asyncio.to_thread(_fetch)


def song_to_search_dict(
    song: ProviderSong, *, processed_song: SongsRow | None = None
) -> SongSearchPayload:
    """Convert a normalized song dict to the SearchSong JSON shape."""
    artist = song.get("primary_artist", {})
    release_date = song.get("release_date_for_display") or ""

    art = song.get("song_art_image_url") or ""
    thumbnail = song.get("song_art_image_thumbnail_url") or art

    return {
        "providerId": song.get("id") or "",
        "provider": "youtube",
        "title": song.get("title") or "",
        "artist": artist.get("name", ""),
        "artistId": artist.get("id", ""),
        "artistThumbnail": artist.get("image_url")
        or artist.get("header_image_url")
        or "",
        "url": song.get("url") or "",
        "art": art,
        "thumbnail": thumbnail,
        "releaseDate": str(release_date),
        "processedSong": processed_song,
    }
