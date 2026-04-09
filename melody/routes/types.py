"""Pydantic response models for all routes."""

from __future__ import annotations

from typing import Literal

from melody.models.database import SongProvider
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ProcessedSong(BaseModel):
    """A song record as stored in the database."""

    model_config = {"extra": "allow"}

    id: int
    provider: SongProvider
    provider_id: str | None = None
    title: str
    artist: str
    status: str


class SongSearchResult(BaseModel):
    """A song entry returned by search and artist-songs endpoints."""

    providerId: str | None = None
    provider: SongProvider
    title: str
    artist: str
    artistId: str
    artistThumbnail: str
    url: str
    art: str
    thumbnail: str
    releaseDate: str
    processedSong: ProcessedSong | None = None


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


class HomeResponse(BaseModel):
    songs: list[SongSearchResult]


# ---------------------------------------------------------------------------
# Lyrics
# ---------------------------------------------------------------------------


class LyricsWord(BaseModel):
    word: str
    start: float | None
    end: float | None
    score: float | None = None
    singer: str | None = None


class LyricsSegment(BaseModel):
    id: int | None = None
    start: float | None
    end: float | None
    text: str
    words: list[LyricsWord] = Field(default_factory=list)
    singer: str | None = None


class SyncedLyrics(BaseModel):
    """Word-level timed lyrics produced by the alignment pipeline."""

    model_config = {"extra": "allow"}

    segments: list[LyricsSegment] = Field(default_factory=list)


class PlainLyrics(BaseModel):
    """Plain-text lyrics from LRCLIB (no word-level timing)."""

    text: str


LyricsSource = Literal["whisper", "ctc", "human"]


class LyricsData(BaseModel):
    """Lyrics payload shared by /lyrics and karaoke experience responses."""

    text: str
    segments: list[LyricsSegment] | None = None
    language: str | None = None
    source: LyricsSource | None = None


class LyricsResponse(BaseModel):
    source: str
    lyrics: LyricsData


# ---------------------------------------------------------------------------
# Karaoke experience
# ---------------------------------------------------------------------------


class KaraokeExperience(BaseModel):
    """Full karaoke experience returned after processing."""

    song: ProcessedSong
    leadVocalsUri: str
    backingVocalsUri: str
    backgroundTrackUri: str
    lyrics: LyricsData
    leadVocalsWaveformUri: str
    backingVocalsWaveformUri: str
    backgroundTrackWaveformUri: str
