"""Karaoke generation and retrieval routes."""

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from melody.pipelines import process_lyrics
from melody.models import separation
from melody.routes.types import KaraokeExperience
from melody.services import music_provider as music_provider_svc
from melody.services import lrclib
from melody.services import supabase_client as db
from melody.services import youtube as yt
from melody.services.config import TEMP_DIR
from melody.services.experience import build_experience_from_db, prepare_and_distribute
from melody.services.gemini import assign_singers

router = APIRouter(prefix="/karaoke", tags=["karaoke"])


@dataclass
class KaraokeGenerationStats:
    """Tracks timing for each stage of karaoke generation."""

    provider_metadata: float | None = None
    download_and_lyrics: float | None = None
    demucs_separation: float = 0.0
    lyrics_alignment: float = 0.0
    singer_assignment: float | None = None
    encode_and_waveform: float | None = None
    file_uploads: float | None = None
    total: float = 0.0

    def print_summary(self, song_id: int) -> None:
        """Print formatted statistics summary to console."""
        print("\n" + "=" * 60)
        print(f"Karaoke Generation Statistics - Song ID: {song_id}")
        print("=" * 60)
        if self.provider_metadata is not None:
            print(f"  Provider metadata fetch:  {self.provider_metadata:6.2f}s")
        if self.download_and_lyrics is not None:
            print(f"  Audio download + lyrics:  {self.download_and_lyrics:6.2f}s")
        print(f"  Stems separation:        {self.demucs_separation:6.2f}s")
        print(f"  Lyrics alignment:         {self.lyrics_alignment:6.2f}s")
        if self.singer_assignment is not None:
            print(f"  Singer assignment:        {self.singer_assignment:6.2f}s")
        if self.encode_and_waveform is not None:
            print(f"  Encoding + waveforms:     {self.encode_and_waveform:6.2f}s")
        if self.file_uploads is not None:
            print(f"  File uploads:             {self.file_uploads:6.2f}s")
        print("-" * 60)
        print(f"  TOTAL:                    {self.total:6.2f}s")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# GET /karaoke/song/{song_id}
# ---------------------------------------------------------------------------


@router.get("/song/{song_id}")
async def get_karaoke_by_id(song_id: int) -> KaraokeExperience:
    song = await db.get_song_by_id(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail=f"Song {song_id} not found.")
    if song["status"] == "failed":
        return await _generate_provider_experience(
            provider_id=song["provider_id"], song_id=song_id
        )
    elif song["status"] != "done":
        raise HTTPException(
            status_code=202, detail="Karaoke experience is still being processed."
        )
    return await build_experience_from_db(song)


# ---------------------------------------------------------------------------
# GET /karaoke/provider/{provider_id}
# ---------------------------------------------------------------------------


@router.get("/provider/{provider_id}")
async def get_karaoke_by_provider_id(provider_id: str) -> KaraokeExperience:
    song = await db.get_song_by_provider("youtube", provider_id)

    if song is not None:
        if song["status"] == "done":
            return await build_experience_from_db(song)
        if song["status"] == "processing":
            raise HTTPException(
                status_code=202,
                detail="Karaoke experience is still being processed.",
            )
        # failed → regenerate

    song_id = song["id"] if song else None
    return await _generate_provider_experience(provider_id, song_id)


async def _generate_provider_experience(
    provider_id: str, song_id: int | None
) -> KaraokeExperience:
    start_time = time.time()
    stats = KaraokeGenerationStats()

    # Fetch provider metadata + start YouTube download + LRCLIB lyrics in parallel
    t0 = time.time()
    provider_future = music_provider_svc.get_song(provider_id)

    provider_song, media = await provider_future
    stats.provider_metadata = time.time() - t0
    youtube_url = next(
        (m["url"] for m in media if m.get("provider") == "youtube"), None
    )
    if youtube_url is None:
        raise HTTPException(
            status_code=422, detail="No YouTube media found for this song."
        )

    artist = (provider_song.get("primary_artist") or {}).get("name", "Unknown Artist")
    title = provider_song.get("title") or "Unknown Title"

    # Download audio + fetch lyrics in parallel
    t0 = time.time()
    download_future = yt.download_audio(youtube_url)
    lyrics_future = lrclib.fetch_lyrics(artist_name=artist, track_name=title)

    # Upsert DB record
    song_row = await db.upsert_song(
        song_id=song_id,
        provider="youtube",
        provider_id=provider_id,
        title=title,
        artist=artist,
        status="processing",
    )
    db_song_id: int = song_row["id"]

    try:
        (audio_path, duration), lyrics = await asyncio.gather(
            download_future, lyrics_future
        )
        stats.download_and_lyrics = time.time() - t0
        return await _run_pipeline(
            db_song_id,
            audio_path,
            lyrics,
            stats,
            start_time,
            title=title,
            artist=artist,
        )
    except Exception:
        await db.update_song_status(db_song_id, "failed")
        raise


# ---------------------------------------------------------------------------
# POST /karaoke  (direct file upload)
# ---------------------------------------------------------------------------


@router.post("/", status_code=status.HTTP_200_OK)
async def upload_karaoke(
    audio: UploadFile,
    title: str = "Unknown Title",
    artist: str = "Unknown Artist",
) -> KaraokeExperience:
    os.makedirs(TEMP_DIR, exist_ok=True)
    suffix = Path(audio.filename or "audio.mp3").suffix or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR)
    try:
        tmp.write(await audio.read())
        tmp.close()
        audio_path = tmp.name

        song_row = await db.upsert_song(
            provider="user_upload",
            provider_id=None,
            title=title,
            artist=artist,
            status="processing",
        )
        db_song_id: int = song_row["id"]

        try:
            start_time = time.time()
            stats = KaraokeGenerationStats()
            return await _run_pipeline(
                db_song_id,
                audio_path,
                None,
                stats,
                start_time,
                title=title,
                artist=artist,
            )
        except Exception:
            await db.update_song_status(db_song_id, "failed")
            raise
    finally:
        # audio_path will be cleaned up by prepare_and_distribute; the tmp file
        # may already be gone — ignore errors.
        try:
            os.remove(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Singer-assignment merge helper
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace for fuzzy line matching."""
    return " ".join(text.lower().split())


def _apply_singers(lyrics_result: dict, singer_map: list[dict[str, str]]) -> None:
    """Merge Gemini singer tags into lyrics segments **in-place**.

    ``singer_map`` is a list of ``{"line": ..., "singer": ...}`` dicts
    produced by ``gemini.assign_singers``.  Each segment whose normalized
    text matches a line gets its ``singer`` field set.
    """
    if not singer_map:
        return

    lookup: dict[str, str] = {}
    for entry in singer_map:
        key = _normalize(entry["line"])
        # First occurrence wins (avoids repeated chorus overwriting).
        lookup.setdefault(key, entry["singer"])

    for seg in lyrics_result.get("segments", []):
        seg_key = _normalize(seg.get("text", ""))
        singer = lookup.get(seg_key)
        if singer:
            seg["singer"] = singer


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------


async def _run_pipeline(
    song_id: int,
    audio_path: str,
    lyrics: str | None,
    stats: KaraokeGenerationStats,
    start_time: float,
    *,
    title: str = "Unknown Title",
    artist: str = "Unknown Artist",
) -> KaraokeExperience:
    """Run Demucs + lyrics alignment, then upload and return experience."""
    output_dir = os.path.join(TEMP_DIR, str(song_id))
    os.makedirs(output_dir, exist_ok=True)

    # Kick off Gemini singer assignment in parallel with the heavy model work.
    # It only needs the plain lyrics text, so it can run concurrently.
    singer_future = None
    if lyrics is not None:
        singer_future = asyncio.ensure_future(assign_singers(lyrics, title, artist))

    # Demucs separation — CPU/GPU-bound, runs in the shared thread executor
    # with the global inference semaphore held (see server.py).
    from melody.server import inference_semaphore

    t0 = time.time()
    async with inference_semaphore:
        vocals_path = await run_in_threadpool(
            separation.separate_audio,
            audio_path,
            output_dir,
        )
    stats.demucs_separation = time.time() - t0

    if vocals_path is None:
        raise HTTPException(status_code=500, detail="Audio separation failed.")

    stem_dir = str(Path(vocals_path).parent)
    background_track = os.path.join(stem_dir, "instrumental.wav")

    # Lyrics alignment — CPU-bound, still under semaphore
    t0 = time.time()
    async with inference_semaphore:
        lyrics_result = await run_in_threadpool(process_lyrics, vocals_path, lyrics)
    stats.lyrics_alignment = time.time() - t0

    # If we didn't have lyrics upfront (whisper path), run Gemini now on the
    # transcribed text.
    if singer_future is None:
        transcribed_text = lyrics_result.get("text", "")
        if transcribed_text:
            singer_future = asyncio.ensure_future(
                assign_singers(transcribed_text, title, artist)
            )

    # Await Gemini result and merge singer tags into segments
    if singer_future is not None:
        t0 = time.time()
        singer_map = await singer_future
        stats.singer_assignment = time.time() - t0
        _apply_singers(lyrics_result, singer_map)

    # Write karaoke.json
    import json as _json

    karaoke_data = {
        "status": "success",
        "input": audio_path,
        "tracks": {
            "vocals": vocals_path,
            "instrumental": background_track,
        },
        "lyrics": lyrics_result,
        "metadata": {
            "segments_count": len(lyrics_result.get("segments", [])),
            "words_count": sum(
                len(seg.get("words", [])) for seg in lyrics_result.get("segments", [])
            ),
        },
    }

    karaoke_json_path = os.path.join(output_dir, "karaoke.json")
    with open(karaoke_json_path, "w", encoding="utf-8") as f:
        _json.dump(karaoke_data, f, ensure_ascii=False, indent=2)

    result = await prepare_and_distribute(
        song_id,
        karaoke_json_path,
        vocals_path,
        background_track,
        stats,
    )

    # Print statistics
    stats.total = time.time() - start_time
    stats.print_summary(song_id)

    return result
