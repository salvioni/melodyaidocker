<h3 align="center">melody.ai</h3>
<p align="center">your karaoke platform</p>

AI karaoke generation pipeline for melody.io. Separates audio, generates/aligns lyrics, and produces word-level timing for karaoke experiences.

## Installation

### - Docker (recommended)

The easiest way to run melody.ai. No need to install Python, ffmpeg, or any other dependency.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

> The server requires at least **8 GB of RAM** to load all three AI models.

```bash
git clone https://github.com/themelodyai/melody.ai.git
cd melody.ai
docker compose up
```

Wait for the models to load (~60 seconds on first run), then check readiness:

```bash
curl http://localhost:8001/health
# {"status":"ok","models_loaded":true}
```



#### Optional GPU support (NVIDIA only)

1. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the `deploy` block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

#### Useful commands

```bash
docker compose up             # start
docker compose down           # stop
docker compose logs -f        # follow logs in real time
```

---

### - Manual installation

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), ffmpeg, audiowaveform

```bash
# macOS
brew install ffmpeg audiowaveform

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-base.txt
```

#### macOS

```bash
uv pip install -r requirements-mac.txt
```

#### NVIDIA GPUs

On CUDA machines, melody.ai now auto-selects the Demucs separator backend so the pipeline can use PyTorch CUDA without extra code changes.

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
melody.ai/
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── .dockerignore             # Files excluded from Docker build
├── requirements-base.txt     # Core Python dependencies
├── requirements-mac.txt      # macOS-specific dependencies
├── requirements-windows.txt  # Windows/CUDA-specific dependencies
├── tests.py                  # Test suite
└── melody/
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
```
