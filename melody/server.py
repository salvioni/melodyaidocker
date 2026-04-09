"""FastAPI server for melody.ai.

All three AI models are loaded once at startup and kept in memory for the
lifetime of the process.  A single asyncio.Semaphore ensures only one
GPU/MPS inference runs at a time; other requests queue and wait.

Start with:
    python -m uvicorn melody.server:app --host 0.0.0.0 --port 8001 --workers 1

Use --workers 1 — multiple workers would each load the models, exhausting GPU
memory. Concurrency within a single worker is handled by the semaphore + async.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from melody.models import ctc_alignment, separation, whisper_alignment
from melody.routes import catalog, karaoke

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global inference semaphore (GPU is a serial resource)
# ---------------------------------------------------------------------------

#: Acquire this before running any model inference.
inference_semaphore = asyncio.Semaphore(1)

# ---------------------------------------------------------------------------
# Startup: pre-load all models
# ---------------------------------------------------------------------------

_models_loaded = False


def _preload_all_models() -> None:
    global _models_loaded
    t0 = time.time()
    logger.info("Pre-loading AI models…")

    separation.preload()
    logger.info("  Separation backend ready (%.1fs)", time.time() - t0)

    t1 = time.time()
    ctc_alignment._get_mms_fa()
    logger.info("  MMS_FA ready (%.1fs)", time.time() - t1)

    t2 = time.time()
    whisper_alignment._get_model("turbo")
    logger.info("  Whisper turbo ready (%.1fs)", time.time() - t2)

    _models_loaded = True
    logger.info("All models loaded in %.1fs total.", time.time() - t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models in a thread so we don't block the event loop during startup
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _preload_all_models)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="melody.ai",
    description="AI-powered karaoke generation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(karaoke.router)
app.include_router(catalog.router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models_loaded": _models_loaded}


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )
