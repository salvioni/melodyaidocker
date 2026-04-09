import os

from melody.models import separation_demucs as demucs
from melody.models import separation_uvr as uvr
from melody.utils import detect_device


def resolve_backend(preferred: str | None = None) -> str:
    """Resolve the separation backend for the current runtime.

    Default behavior keeps UVR on Apple Silicon / CPU, but switches to Demucs
    on CUDA because this repository's Demucs path already uses PyTorch CUDA
    directly while UVR needs an NVIDIA-specific ONNX Runtime package.
    """
    configured = (
        (preferred or os.getenv("MELODY_SEPARATION_BACKEND", "auto")).strip().lower()
    )
    if configured not in {"auto", "demucs", "uvr"}:
        raise ValueError("MELODY_SEPARATION_BACKEND must be one of: auto, demucs, uvr")

    if configured != "auto":
        return configured

    return "demucs" if detect_device() == "cuda" else "uvr"


def preload() -> None:
    backend = resolve_backend()
    if backend == "demucs":
        demucs.preload_demucs()
    elif backend == "uvr":
        uvr.preload_uvr()


def separate_audio(audio_path: str, output_dir: str = "outputs") -> str | None:
    backend = resolve_backend()
    print(f"Separating audio using backend: {backend}")
    if backend == "uvr":
        try:
            return uvr.separate_audio(audio_path, output_dir)
        except Exception as e:
            print(f"UVR backend unavailable ({e}), falling back to Demucs.")

    return demucs.separate_audio(audio_path, output_dir)
