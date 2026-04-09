"""Utility modules for device detection and logging."""

import logging
import os


def detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    override = os.getenv("MELODY_DEVICE", "").strip().lower()
    if override:
        if override in {"auto", ""}:
            override = ""
        elif override in {"cpu", "cuda", "mps"}:
            return override
        else:
            raise ValueError("MELODY_DEVICE must be one of: auto, cpu, cuda, mps")

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup_logging(level=logging.INFO, json_mode=False):
    """Configure logging with optional JSON-mode suppression."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if json_mode:
        logging.disable(logging.CRITICAL)
    return logging.getLogger(__name__)
