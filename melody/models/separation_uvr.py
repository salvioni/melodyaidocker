import os
import shutil
import time
from audio_separator.separator import Separator

from melody.utils import detect_device as _detect_device

# Module-level cache for the Separator instance
_cached_separator = None
_cached_model_name: str | None = None

# A fixed, safe directory for the AI engine to do its work
_shared_output_dir = "/tmp/uvr_shared_output"


def _create_separator() -> Separator:
    return Separator(
        output_dir=_shared_output_dir,
        use_autocast=True,
        mdxc_params={
            "segment_size": 512,
            "override_model_segment_size": True,
            "batch_size": 4,
            "overlap": 2,
            "pitch_shift": 0,
        },
        mdx_params={
            "hop_length": 1024,
            "segment_size": 1024,
            "overlap": 0.02,
            "batch_size": 8,
            "enable_denoise": False,
        },
    )


def preload_uvr(model_name: str = "UVR-MDX-NET-Inst_HQ_3.onnx") -> None:
    """Load the UVR model into the module-level cache."""
    global _cached_separator, _cached_model_name

    device = _detect_device()

    print(f"Pre-loading UVR model '{model_name}' with preferred device {device}...")
    t0 = time.time()

    os.makedirs(_shared_output_dir, exist_ok=True)

    # We lock the Separator to a single shared directory upon initialization
    separator = _create_separator()
    separator.load_model(model_name)

    _cached_separator = separator
    _cached_model_name = model_name
    print(f"UVR model ready in {time.time() - t0:.1f}s")


def separate_audio(
    audio_path: str,
    output_dir: str = "outputs",
    model: str = "UVR-MDX-NET-Inst_HQ_3.onnx",
) -> str | None:
    """
    Separate an audio file into stems using cached UVR models.
    """
    global _cached_separator, _cached_model_name

    if not os.path.isfile(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        return None

    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, "uvr", audio_name)

    final_vocals_path = os.path.join(stem_dir, "vocals.wav")
    final_instrumental_path = os.path.join(stem_dir, "instrumental.wav")

    if os.path.isfile(final_vocals_path) and os.path.isfile(final_instrumental_path):
        print(f"Skipping separation, cached output found: {stem_dir}")
        return final_vocals_path

    os.makedirs(stem_dir, exist_ok=True)
    start = time.time()

    # Reuse the cached separator if available, otherwise create it
    if _cached_separator is not None and _cached_model_name == model:
        separator = _cached_separator
    else:
        print("Cache miss. Initializing Audio Separator locally...")
        os.makedirs(_shared_output_dir, exist_ok=True)
        separator = _create_separator()
        separator.load_model(model)
        _cached_separator = separator
        _cached_model_name = model

    print(f"Separating '{audio_name}'...")
    try:
        # Returns a list of the generated filenames: e.g. ["song_(Vocals).wav", "song_(Instrumental).wav"]
        output_files = separator.separate(audio_path)
    except Exception as e:
        print(f"Separation failed: {e}")
        return None

    # THE FIX: Move files from the shared workspace into your specific stem_dir
    for file in output_files:
        old_path = os.path.join(_shared_output_dir, file)

        if "(Vocals)" in file:
            shutil.move(old_path, final_vocals_path)
        elif "(Instrumental)" in file:
            shutil.move(old_path, final_instrumental_path)

    elapsed = time.time() - start
    print(f"Separation completed in {elapsed:.1f}s")
    print(f"Stems saved to: {stem_dir}")

    if os.path.isfile(final_vocals_path):
        return final_vocals_path

    return None
