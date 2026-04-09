<h3 align="center">melody.ai</h1>
<p align="center">your karaoke platform</p>

AI karaoke generation pipeline for melody.io. Separates audio, generates/aligns lyrics, and produces word-level timing for karaoke experiences.

## Installation

**Prerequisites:** Python 3.12, ffmpeg, audiowaveform

```bash
# macOS
brew install ffmpeg audiowaveform

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-base.txt
```

### macOS

On macOS, install the dependencies:

```bash
uv pip install -r requirements-mac.txt
```

### NVIDIA GPUs

On CUDA machines, melody.ai now auto-selects the Demucs separator backend so the
pipeline can use PyTorch CUDA without extra code changes.

```bash
uv pip install -r requirements-windows.txt
```

## Running the server

```bash
python -m uvicorn melody.server:app --host 0.0.0.0 --port 8001 --workers 1 --reload
```

> **`--workers 1` is required.** Multiple workers would each load the AI models into GPU memory independently, likely exhausting available memory.

On first start the server pre-loads all three models (Demucs, MMS_FA, Whisper turbo) before accepting requests. This takes ~30–60 s once; every subsequent request reuses the in-memory models.

Check readiness:
```bash
curl http://localhost:8001/health
# {"status":"ok","models_loaded":true}
```

### Environment variables

| Variable                    | Default       | Description                               |
| --------------------------- | ------------- | ----------------------------------------- |
| `SUPABASE_URL`              | *(bundled)*   | Supabase project URL                      |
| `SUPABASE_SERVICE_ROLE_KEY` | *(bundled)*   | Supabase service role key                 |
| `TEMP_DIR`                  | `/tmp/melody` | Local directory for temporary audio files |
| `MELODY_DEVICE`             | `auto`        | Force `cuda`, `mps`, or `cpu`             |
| `MELODY_SEPARATION_BACKEND` | `auto`        | Force `demucs` or `uvr`                   |

Override any variable before starting the server:
```bash
TEMP_DIR=/data/tmp python -m uvicorn melody.server:app --port 8001 --workers 1
```

## API

All endpoints return JSON. The response shapes are compatible with `melody.io`.

| Method | Path                              | Description                                 |
| ------ | --------------------------------- | ------------------------------------------- |
| `GET`  | `/health`                         | Server and model readiness                  |
| `GET`  | `/home`                           | All processed songs                         |
| `GET`  | `/search/{query}`                 | Search YouTube Music for songs              |
| `GET`  | `/artist/{artist_id}/songs`       | Artist song catalogue                       |
| `GET`  | `/karaoke/song/{song_id}`         | Retrieve karaoke by database ID             |
| `GET`  | `/karaoke/provider/{provider_id}` | Retrieve or generate karaoke by provider ID |
| `POST` | `/karaoke`                        | Upload an audio file and generate karaoke   |
| `GET`  | `/lyrics/{song_id}`               | Word-level lyrics for a processed song      |
| `GET`  | `/lyrics/provider/{provider_id}`  | Plain/synced lyrics from LRCLIB             |

Interactive docs are available at `http://localhost:8001/docs` while the server is running.

## Technologies

**Separation:** Demucs (Meta) — audio source separation
**Lyrics Transcription:** Whisper turbo (OpenAI) — speech-to-text
**Lyrics Alignment:** MMS_FA (Meta) + uroman — CTC forced alignment
**Devices:** Auto-detects CUDA › MPS › CPU
**Separation backend:** Auto-selects Demucs on CUDA, UVR elsewhere
**Server:** FastAPI + uvicorn
**Storage:** Supabase (PostgreSQL + S3-compatible storage)

## Project Structure

```
melody/
├── server.py                 # FastAPI app, lifespan, inference semaphore
├── models/                   # AI model wrappers
│   ├── separation.py         # Demucs (with startup pre-load cache)
│   ├── ctc_alignment.py      # MMS_FA forced alignment
│   └── whisper_alignment.py  # Whisper transcription
├── pipelines/
│   └── __init__.py           # process_lyrics() orchestration
├── routes/                   # FastAPI routers
│   ├── karaoke.py
│   ├── search.py
│   ├── lyrics.py
│   ├── artist.py
│   └── home.py
├── services/                 # External integrations
│   ├── config.py             # Env vars
│   ├── supabase_client.py    # Database and storage
│   ├── music_provider.py     # YouTube Music API (ytmusicapi)
│   ├── lrclib.py             # LRCLIB lyrics API
│   ├── youtube.py            # yt-dlp download
│   ├── waveform.py           # audiowaveform CLI wrapper
│   └── experience.py         # KaraokeExperience assembly
└── utils/
    ├── device.py             # Device detection
    └── logging.py            # Logging setup

tests.py                      # Test suite
```

