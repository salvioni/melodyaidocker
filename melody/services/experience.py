"""Orchestrates karaoke experience preparation and distribution."""

import asyncio
import json
import os
import subprocess
import time
from typing import TYPE_CHECKING

from melody.models.database import SongsRow
from melody.routes.types import KaraokeExperience, LyricsData, ProcessedSong
from melody.services import supabase_client as db
from melody.services.waveform import generate_waveform

if TYPE_CHECKING:
    from melody.routes.karaoke import KaraokeGenerationStats


async def _encode_for_delivery(wav_path: str) -> str:
    """Re-encode a WAV file to AAC 128 kbps (.m4a) for fast client downloads.

    AAC 128 kbps provides good quality for vocals while keeping files small
    (~20–30× smaller than WAV). Faster uploads and downloads.
    """
    out = os.path.splitext(wav_path)[0] + ".m4a"
    await asyncio.to_thread(
        subprocess.run,
        ["ffmpeg", "-i", wav_path, "-c:a", "aac", "-b:a", "128k", "-y", out],
        check=True,
        capture_output=True,
    )
    return out


async def prepare_and_distribute(
    song_id: int,
    karaoke_json_path: str,
    lead_vocals_path: str,
    background_track_path: str,
    stats: "KaraokeGenerationStats",
) -> KaraokeExperience:
    """Upload all karaoke assets to Supabase, then return a KaraokeExperience.

    Steps (as parallel as possible):
      1. Encode both WAV stems to AAC and generate both waveforms — all 4 in parallel.
      2. Upload all 5 files to Supabase Storage in parallel.
      3. Update song status to "done" and create signed URLs in parallel.
      4. Return the KaraokeExperience.
    """
    # Step 1 — encode + waveform generation all in parallel
    t0 = time.time()
    (
        lead_vocals_m4a,
        background_track_m4a,
        lead_waveform_path,
        bg_waveform_path,
    ) = await asyncio.gather(
        _encode_for_delivery(lead_vocals_path),
        _encode_for_delivery(background_track_path),
        generate_waveform(lead_vocals_path),
        generate_waveform(background_track_path),
    )
    stats.encode_and_waveform = time.time() - t0

    # Step 2 — upload everything in parallel
    t0 = time.time()
    upload_paths = [
        f"{song_id}/lead_vocals.m4a",
        f"{song_id}/background_track.m4a",
        f"{song_id}/lead_vocals_waveform.json",
        f"{song_id}/background_track_waveform.json",
        f"{song_id}/karaoke.json",
    ]
    await asyncio.gather(
        db.upload_file(upload_paths[0], lead_vocals_m4a),
        db.upload_file(upload_paths[1], background_track_m4a),
        db.upload_file(upload_paths[2], lead_waveform_path),
        db.upload_file(upload_paths[3], bg_waveform_path),
        db.upload_file(upload_paths[4], karaoke_json_path),
    )
    stats.file_uploads = time.time() - t0

    # Step 3 — DB update + signed URLs in parallel
    audio_paths = [p for p in upload_paths if not p.endswith("karaoke.json")]
    song_future = db.update_song_status(song_id, "done")
    urls_future = db.create_signed_urls(audio_paths)

    song, signed_urls = await asyncio.gather(song_future, urls_future)

    # Read lyrics from karaoke.json
    with open(karaoke_json_path, encoding="utf-8") as f:
        karaoke_data = json.load(f)
    lyrics = karaoke_data.get("lyrics", {})

    # Clean up local temp files (fire-and-forget)
    for path in [
        lead_vocals_path,
        background_track_path,
        lead_vocals_m4a,
        background_track_m4a,
        lead_waveform_path,
        bg_waveform_path,
        karaoke_json_path,
    ]:
        try:
            os.remove(path)
        except OSError:
            pass

    return _build_experience(song, lyrics, signed_urls)


async def build_experience_from_db(song: SongsRow) -> KaraokeExperience:
    """Reconstruct a KaraokeExperience for an already-processed song."""
    song_id = song["id"]
    audio_paths = [
        f"{song_id}/lead_vocals.m4a",
        f"{song_id}/background_track.m4a",
        f"{song_id}/lead_vocals_waveform.json",
        f"{song_id}/background_track_waveform.json",
    ]

    urls_future = db.create_signed_urls(audio_paths)
    karaoke_future = db.download_karaoke_json(song_id)

    signed_urls, karaoke_data = await asyncio.gather(urls_future, karaoke_future)
    lyrics = karaoke_data.get("lyrics", {})

    return _build_experience(song, lyrics, signed_urls)


def _build_experience(
    song: SongsRow, lyrics: dict, signed_urls: list[str]
) -> KaraokeExperience:
    lead_vocals_url, background_track_url, lead_waveform_url, bg_waveform_url = (
        signed_urls[0],
        signed_urls[1],
        signed_urls[2],
        signed_urls[3],
    )
    return KaraokeExperience(
        song=ProcessedSong.model_validate(song),
        leadVocalsUri=lead_vocals_url,
        backingVocalsUri=lead_vocals_url,  # no backing vocals yet
        backgroundTrackUri=background_track_url,
        lyrics=LyricsData.model_validate(lyrics),
        leadVocalsWaveformUri=lead_waveform_url,
        backingVocalsWaveformUri=bg_waveform_url,
        backgroundTrackWaveformUri=bg_waveform_url,
    )
