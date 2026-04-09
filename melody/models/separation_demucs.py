import subprocess
import sys
import os
import time

from melody.utils import detect_device as _detect_device

# Module-level Demucs model cache — populated by preload_demucs() at server
# startup so the model is ready before the first request arrives.
_cached_model = None
_cached_model_name: str | None = None
_cached_device: str | None = None


def preload_demucs(model_name: str = "mdx_q", device: str = "auto") -> None:
    """Load the Demucs model into the module-level cache.

    Call this once at server startup.  Subsequent calls to ``separate_audio``
    will reuse the cached model, skipping the ~10-15 s load time per request.
    """
    global _cached_model, _cached_model_name, _cached_device
    from demucs.pretrained import get_model

    if device == "auto":
        device = _detect_device()
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"Pre-loading Demucs model '{model_name}' on {device}…")
    t0 = time.time()
    _cached_model = get_model(model_name)
    _cached_model.to(device)
    _cached_model_name = model_name
    _cached_device = device
    print(f"Demucs model ready in {time.time() - t0:.1f}s")


def _separate_via_api(
    audio_path: str, stem_dir: str, device: str, model_name: str
) -> bool:
    """Separate audio using the Demucs Python API. Returns True on success."""
    import pathlib
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile, save_audio

    # Reuse the pre-loaded model when available; fall back to loading on demand.
    if (
        _cached_model is not None
        and _cached_model_name == model_name
        and _cached_device == device
    ):
        model = _cached_model
    else:
        model = get_model(model_name)
        model.to(device)

    audio_file = AudioFile(pathlib.Path(audio_path))
    wav = audio_file.read(
        streams=0, samplerate=model.samplerate, channels=model.audio_channels
    )
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    # Separate
    sources = apply_model(model, wav[None], device=device, progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    os.makedirs(stem_dir, exist_ok=True)
    source_dict = dict(zip(model.sources, sources))
    for name, source in source_dict.items():
        save_audio(source, os.path.join(stem_dir, f"{name}.wav"), model.samplerate)

    # Generate instrumental in-memory from tensors already in hand — no disk reload
    if all(k in source_dict for k in ("drums", "bass", "other")):
        instrumental = source_dict["drums"] + source_dict["bass"] + source_dict["other"]
        save_audio(
            instrumental,
            os.path.join(stem_dir, "instrumental.wav"),
            model.samplerate,
        )

    return True


def _separate_via_cli(
    audio_path: str, output_dir: str, device: str, model: str
) -> bool:
    """Fallback: separate audio using the Demucs CLI. Returns True on success."""
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "-d",
        device,
        "-o",
        output_dir,
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Demucs CLI error: {result.stderr}")
        return False
    return True


def _generate_instrumental(stem_dir: str) -> None:
    """Generate an instrumental track by summing drums + bass + other stems."""
    try:
        import torchaudio

        stems = []
        for name in ("drums", "bass", "other"):
            path = os.path.join(stem_dir, f"{name}.wav")
            if not os.path.isfile(path):
                return
            waveform, sr = torchaudio.load(path)
            stems.append((waveform, sr))

        samplerate = stems[0][1]
        instrumental = sum(s[0] for s in stems)
        torchaudio.save(
            uri=os.path.join(stem_dir, "instrumental.wav"),
            src=instrumental,
            sample_rate=samplerate,
        )
    except Exception as e:
        print(f"Could not generate instrumental track: {e}")


def separate_audio(
    audio_path: str,
    output_dir: str = "outputs",
    device: str = "auto",
    model: str = "htdemucs",
) -> str | None:
    """
    Separate an audio file into stems using Demucs.

    Produces: vocals.wav, drums.wav, bass.wav, other.wav, instrumental.wav

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory where stems will be saved.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        model: Demucs model name.

    Returns:
        Path to the vocals WAV file, or None on failure.
    """
    if not os.path.isfile(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        return None

    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    stem_dir = os.path.join(output_dir, model, audio_name)
    vocals_path = os.path.join(stem_dir, "vocals.wav")

    if os.path.isfile(vocals_path):
        print(f"Skipping separation, cached output found: {stem_dir}")
        return vocals_path

    if device == "auto":
        device = _detect_device()
    print(f"Using device: {device}")

    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    start = time.time()
    separated = False
    # Tracks whether _separate_via_api already generated instrumental.wav in-memory.
    # The CLI fallback does not, so _generate_instrumental must be called for it.
    instrumental_generated = False

    try:
        separated = _separate_via_api(audio_path, stem_dir, device, model)
        instrumental_generated = separated
    except ImportError:
        print("Demucs API not available, trying CLI fallback...")
        try:
            separated = _separate_via_cli(audio_path, output_dir, device, model)
        except Exception as e:
            print(f"CLI fallback failed: {e}")
    except RuntimeError as e:
        if device == "mps":
            print(f"MPS failed ({e}), retrying on CPU...")
            try:
                separated = _separate_via_api(audio_path, stem_dir, "cpu", model)
                instrumental_generated = separated
            except Exception as e2:
                print(f"CPU retry failed: {e2}")
        else:
            print(f"Separation failed: {e}")
    except Exception as e:
        print(f"Separation failed: {e}")

    if not separated:
        print("Error: audio separation failed.")
        print("Make sure demucs is installed: pip install demucs")
        return None

    if not instrumental_generated:
        _generate_instrumental(stem_dir)

    elapsed = time.time() - start
    print(f"Separation completed in {elapsed:.1f}s")
    print(f"Stems saved to: {stem_dir}")

    if os.path.isfile(vocals_path):
        return vocals_path

    print(f"Error: expected vocals file not found: {vocals_path}")
    return vocals_path
