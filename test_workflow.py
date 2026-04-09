"""End-to-end API workflow tests.

Simulates the real user journey through the melody.ai API:
  1. Health check
  2. Search for a song
  3. Browse artist songs
  4. Preview lyrics (LRCLIB)
  5. Generate a karaoke experience (full pipeline)
  6. Retrieve the karaoke experience by song ID
  7. Get synced lyrics for a processed song
  8. Browse the home page
  9. Upload audio directly for karaoke generation

All external services (YouTube Music, YouTube, Supabase, LRCLIB) and heavy
AI model inference are mocked so the tests run fast without network
access, GPU, or API keys.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures: shared mock data
# ---------------------------------------------------------------------------

MOCK_GENIUS_SONG = {
    "id": 12345,
    "title": "Kill This Love",
    "primary_artist": {
        "id": 678,
        "name": "BLACKPINK",
        "image_url": "https://example.com/bp_thumb.jpg",
    },
    "url": "https://genius.com/Blackpink-kill-this-love-lyrics",
    "song_art_image_url": "https://example.com/ktl_art.jpg",
    "song_art_image_thumbnail_url": "https://example.com/ktl_thumb.jpg",
    "release_date_for_display": "April 5, 2019",
}

MOCK_GENIUS_MEDIA = [
    {"provider": "youtube", "url": "https://www.youtube.com/watch?v=FAKE_ID"},
    {"provider": "spotify", "url": "https://open.spotify.com/track/fake"},
]

MOCK_LYRICS_TEXT = (
    "Yeah yeah yeah\nYeah yeah yeah\nWe all commit to love\nThat makes you cry oh\n"
)

MOCK_LYRICS_RESULT = {
    "source": "ctc",
    "text": MOCK_LYRICS_TEXT,
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 2.5,
            "text": "Yeah yeah yeah",
            "words": [
                {"word": "Yeah", "start": 0.0, "end": 0.5, "score": 0.9},
                {"word": "yeah", "start": 0.6, "end": 1.1, "score": 0.9},
                {"word": "yeah", "start": 1.2, "end": 1.7, "score": 0.9},
            ],
        },
        {
            "id": 1,
            "start": 2.5,
            "end": 5.0,
            "text": "Yeah yeah yeah",
            "words": [
                {"word": "Yeah", "start": 2.5, "end": 3.0, "score": 0.9},
                {"word": "yeah", "start": 3.1, "end": 3.6, "score": 0.9},
                {"word": "yeah", "start": 3.7, "end": 4.2, "score": 0.9},
            ],
        },
    ],
}

MOCK_DB_SONG = {
    "id": 1,
    "provider": "youtube",
    "provider_id": "12345",
    "title": "Kill This Love",
    "artist": "BLACKPINK",
    "status": "done",
}

MOCK_KARAOKE_JSON = {
    "status": "success",
    "lyrics": MOCK_LYRICS_RESULT,
    "tracks": {"vocals": "/tmp/vocals.wav", "instrumental": "/tmp/instrumental.wav"},
    "metadata": {"segments_count": 2, "words_count": 6},
}

MOCK_EXPERIENCE = {
    "song": MOCK_DB_SONG,
    "leadVocalsUri": "https://example.com/vocals.aac",
    "backingVocalsUri": "https://example.com/backing.aac",
    "backgroundTrackUri": "https://example.com/bg.aac",
    "lyrics": MOCK_LYRICS_RESULT,
    "leadVocalsWaveformUri": "https://example.com/vocals.json",
    "backingVocalsWaveformUri": "https://example.com/backing.json",
    "backgroundTrackWaveformUri": "https://example.com/bg.json",
}


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_genius():
    """Mock music provider API service calls."""
    search = AsyncMock(return_value=[MOCK_GENIUS_SONG])
    get_song = AsyncMock(return_value=(MOCK_GENIUS_SONG, MOCK_GENIUS_MEDIA))
    get_artist_songs = AsyncMock(return_value=[MOCK_GENIUS_SONG])
    song_to_dict = MagicMock(
        side_effect=lambda song, processed_song=None: {
            "providerId": str(song.get("id", "")),
            "provider": "youtube",
            "title": song.get("title", ""),
            "artist": (song.get("primary_artist") or {}).get("name", ""),
            "artistId": str((song.get("primary_artist") or {}).get("id", "")),
            "artistThumbnail": (song.get("primary_artist") or {}).get("image_url", ""),
            "url": song.get("url", ""),
            "art": song.get("song_art_image_url", ""),
            "thumbnail": song.get("song_art_image_thumbnail_url", ""),
            "releaseDate": song.get("release_date_for_display", ""),
            "processedSong": processed_song,
        }
    )

    return {
        "melody.services.music_provider.search_songs": search,
        "melody.services.music_provider.get_song": get_song,
        "melody.services.music_provider.get_artist_songs": get_artist_songs,
        "melody.services.music_provider.song_to_search_dict": song_to_dict,
    }


def _patch_db():
    """Mock Supabase DB + Storage calls."""
    return {
        "melody.services.supabase_client.get_all_songs": AsyncMock(
            return_value=[MOCK_DB_SONG]
        ),
        "melody.services.supabase_client.get_song_by_id": AsyncMock(
            return_value=MOCK_DB_SONG
        ),
        "melody.services.supabase_client.get_song_by_provider": AsyncMock(
            return_value=MOCK_DB_SONG
        ),
        "melody.services.supabase_client.get_songs_by_provider_ids": AsyncMock(
            return_value=[MOCK_DB_SONG]
        ),
        "melody.services.supabase_client.upsert_song": AsyncMock(
            return_value=MOCK_DB_SONG
        ),
        "melody.services.supabase_client.update_song_status": AsyncMock(),
        "melody.services.supabase_client.download_karaoke_json": AsyncMock(
            return_value=MOCK_KARAOKE_JSON
        ),
    }


def _patch_experience():
    """Mock experience builder."""
    return {
        "melody.services.experience.build_experience_from_db": AsyncMock(
            return_value=MOCK_EXPERIENCE
        ),
        "melody.services.experience.prepare_and_distribute": AsyncMock(
            return_value=MOCK_EXPERIENCE
        ),
    }


def _all_patches():
    """Combine all service patches."""
    patches = {}
    patches.update(_patch_genius())
    patches.update(_patch_db())
    patches.update(_patch_experience())
    patches["melody.services.lrclib.fetch_lyrics"] = AsyncMock(
        return_value=MOCK_LYRICS_TEXT
    )
    patches["melody.services.youtube.download_audio"] = AsyncMock(
        return_value=("/tmp/fake_audio.mp3", 180.0)
    )
    patches["melody.models.separation.separate_audio"] = MagicMock(
        return_value="/tmp/vocals.wav"
    )
    patches["melody.pipelines.process_lyrics"] = MagicMock(
        return_value=MOCK_LYRICS_RESULT
    )
    return patches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """Create an async test client with all external services mocked."""
    patches = _all_patches()
    mocks = {target: patch(target, new=mock) for target, mock in patches.items()}

    for m in mocks.values():
        m.start()

    # Import app *after* patching so route-level imports see the mocks.
    # We also skip the heavy model-preloading lifespan.
    from melody.server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    for m in mocks.values():
        m.stop()


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_01_health(client: AsyncClient):
    """User opens the app — first thing is a health / connectivity check."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Search for a song
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_02_search_songs(client: AsyncClient):
    """User types a query in the search bar."""
    resp = await client.get("/search/Kill This Love")
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    assert len(results) >= 1

    song = results[0]
    assert song["title"] == "Kill This Love"
    assert song["artist"] == "BLACKPINK"
    assert song["provider"] == "youtube"
    assert "providerId" in song


# ---------------------------------------------------------------------------
# 3. Browse artist songs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_03_artist_songs(client: AsyncClient):
    """User taps an artist to see more songs."""
    resp = await client.get("/artist/678/songs?per_page=10&page=1")
    assert resp.status_code == 200
    results = resp.json()
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["artistId"] == "678"


# ---------------------------------------------------------------------------
# 4. Preview lyrics (before generating karaoke)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_04_preview_lyrics_from_lrclib(client: AsyncClient):
    """User previews plain lyrics from LRCLIB before committing to generate."""
    resp = await client.get("/lyrics/provider/12345")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "lrclib"
    assert "text" in data["lyrics"]
    assert "Yeah yeah yeah" in data["lyrics"]["text"]


# ---------------------------------------------------------------------------
# 5. Generate karaoke experience (full pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_05_generate_karaoke(client: AsyncClient):
    """User hits 'Generate Karaoke' — triggers the full AI pipeline."""
    # The mock DB says status=done, so it returns the cached experience
    # without re-running the pipeline. This is the happy path when the song
    # was already processed.
    resp = await client.get("/karaoke/provider/12345")
    assert resp.status_code == 200
    data = resp.json()
    assert data["song"]["title"] == "Kill This Love"
    assert data["leadVocalsUri"].startswith("https://")
    assert data["backgroundTrackUri"].startswith("https://")
    assert "segments" in data["lyrics"]


# ---------------------------------------------------------------------------
# 6. Retrieve karaoke by internal song ID
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_06_get_karaoke_by_song_id(client: AsyncClient):
    """User re-opens a previously generated karaoke from their history."""
    resp = await client.get("/karaoke/song/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["song"]["id"] == 1
    assert "leadVocalsUri" in data
    assert "lyrics" in data


# ---------------------------------------------------------------------------
# 7. Get synced lyrics for a processed song
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_07_get_synced_lyrics(client: AsyncClient):
    """User opens the lyrics view for a processed song."""
    resp = await client.get("/lyrics/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "database"
    assert "segments" in data["lyrics"]
    words = data["lyrics"]["segments"][0]["words"]
    assert len(words) > 0
    assert "start" in words[0]
    assert "end" in words[0]


# ---------------------------------------------------------------------------
# 8. Browse home page
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_08_home(client: AsyncClient):
    """User opens the home screen — sees all processed songs."""
    resp = await client.get("/home")
    assert resp.status_code == 200
    data = resp.json()
    assert "songs" in data
    assert isinstance(data["songs"], list)
    assert len(data["songs"]) >= 1
    assert data["songs"][0]["title"] == "Kill This Love"


# ---------------------------------------------------------------------------
# 9. Upload custom audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_09_upload_karaoke(client: AsyncClient):
    """User uploads their own audio file for karaoke generation."""
    fake_audio = io.BytesIO(b"\x00" * 1024)

    resp = await client.post(
        "/karaoke/",
        files={"audio": ("my_song.mp3", fake_audio, "audio/mpeg")},
        data={"title": "My Song", "artist": "Me"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["song"]["title"] == "Kill This Love"  # mocked DB returns this
    assert "leadVocalsUri" in data
    assert "lyrics" in data


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_10_karaoke_song_not_found(client: AsyncClient):
    """Request a non-existent song ID → 404."""
    with patch(
        "melody.services.supabase_client.get_song_by_id",
        new=AsyncMock(return_value=None),
    ):
        resp = await client.get("/karaoke/song/999")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_11_karaoke_still_processing(client: AsyncClient):
    """Song exists but is still processing → 202."""
    processing_song = {**MOCK_DB_SONG, "status": "processing"}
    with patch(
        "melody.services.supabase_client.get_song_by_id",
        new=AsyncMock(return_value=processing_song),
    ):
        resp = await client.get("/karaoke/song/1")
        assert resp.status_code == 202


@pytest.mark.asyncio
async def test_12_lyrics_not_found_on_lrclib(client: AsyncClient):
    """LRCLIB has no lyrics → 404."""
    with patch(
        "melody.services.lrclib.fetch_lyrics",
        new=AsyncMock(return_value=None),
    ):
        resp = await client.get("/lyrics/provider/12345")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_13_generate_karaoke_no_youtube(client: AsyncClient):
    """Song has no YouTube media → 422."""
    no_yt_media = [{"provider": "spotify", "url": "https://open.spotify.com/fake"}]
    failed_song = {**MOCK_DB_SONG, "status": "failed"}
    with (
        patch(
            "melody.services.supabase_client.get_song_by_provider",
            new=AsyncMock(return_value=failed_song),
        ),
        patch(
            "melody.services.music_provider.get_song",
            new=AsyncMock(return_value=(MOCK_GENIUS_SONG, no_yt_media)),
        ),
    ):
        resp = await client.get("/karaoke/provider/12345")
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_14_synced_lyrics_not_found(client: AsyncClient):
    """Karaoke JSON not found for song → 404."""
    with patch(
        "melody.services.supabase_client.download_karaoke_json",
        new=AsyncMock(side_effect=Exception("Not found")),
    ):
        resp = await client.get("/lyrics/1")
        assert resp.status_code == 404
