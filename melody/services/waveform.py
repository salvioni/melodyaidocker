"""Waveform generation using the audiowaveform CLI."""

import asyncio
import subprocess
from pathlib import Path


async def generate_waveform(audio_path: str) -> str:
    """Generate a waveform JSON file for *audio_path*.

    Uses the ``audiowaveform`` CLI tool. Returns the path to the output
    ``.json`` file.

    Requires ``audiowaveform`` to be installed (``brew install audiowaveform``).
    """
    stem = Path(audio_path).stem
    output_path = str(Path(audio_path).with_name(f"{stem}_waveform.json"))

    await asyncio.to_thread(_run_audiowaveform, audio_path, output_path)
    return output_path


def _run_audiowaveform(input_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "audiowaveform",
            "-i",
            input_path,
            "-o",
            output_path,
            "--pixels-per-second",
            "20",
            "--bits",
            "8",
        ],
        check=True,
        capture_output=True,
    )


async def generate_waveforms_parallel(*audio_paths: str) -> list[str]:
    """Generate waveforms for multiple files concurrently."""
    return list(await asyncio.gather(*[generate_waveform(p) for p in audio_paths]))
