import time

from melody.utils import detect_device as _detect_device

_model = None
_model_name = None
_mps_dtw_patched = False
_original_whisper_dtw = None


def _patch_whisper_mps_word_timestamps() -> None:
    """Patch Whisper DTW to avoid float64 on MPS.

    Whisper's DTW path calls x.double() before moving to CPU. MPS does not
    support float64 tensors, so for MPS we force float32 before CPU DTW.
    """
    global _mps_dtw_patched, _original_whisper_dtw
    if _mps_dtw_patched:
        return

    import torch
    import whisper.timing as timing

    if _original_whisper_dtw is None:
        _original_whisper_dtw = timing.dtw

    def _dtw_safe(x: torch.Tensor):
        if x.device.type == "mps":
            return timing.dtw_cpu(x.float().cpu().numpy())
        if _original_whisper_dtw is not None:
            return _original_whisper_dtw(x)
        return timing.dtw_cpu(x.double().cpu().numpy())

    timing.dtw = _dtw_safe
    _mps_dtw_patched = True


def _get_model(name: str = "turbo", device: str | None = None):
    """Load and cache the Whisper model."""
    global _model, _model_name
    import whisper

    if _model is not None and _model_name == name:
        return _model

    if device is None:
        device = _detect_device()

    if device == "mps":
        _patch_whisper_mps_word_timestamps()

    print(f"Loading Whisper model on {device}...")
    start = time.time()
    _model = whisper.load_model(name, device=device)
    _model_name = name
    elapsed = time.time() - start
    print(f"Whisper model loaded in {elapsed:.1f}s on {device}")
    return _model


def transcribe_lyrics(
    audio_path: str,
    model_name: str = "turbo",
    device: str | None = None,
) -> dict:
    """Transcribe lyrics from an audio file using Whisper.

    Use this when lyrics are not available and need to be generated
    from the audio.

    Args:
        audio_path: Path to the audio file (ideally isolated vocals).
        model_name: Whisper model to use.
        device: Device to use (None for auto-detect).

    Returns:
        dict with keys: "text", "language", "segments".
        Each segment contains "words" with "word", "start", "end".
    """
    if device is None:
        device = _detect_device()

    model = _get_model(model_name, device)

    print("Transcribing audio...")
    start = time.time()
    transcribe_kwargs: dict = {"word_timestamps": True}
    if device == "mps":
        transcribe_kwargs["fp16"] = False
    result: dict = model.transcribe(audio_path, **transcribe_kwargs)  # type: ignore
    elapsed = time.time() - start
    print(f"Transcription completed in {elapsed:.1f}s")

    return result


def align_lyrics(
    audio_path: str,
    lyrics: str,
    model_name: str = "turbo",
    device: str | None = None,
) -> dict:
    """Align known lyrics with an audio file using Whisper.

    Whisper uses the provided lyrics as an initial prompt to guide
    transcription, producing word-level timestamps that match the
    known text.

    Args:
        audio_path: Path to the audio file (ideally isolated vocals).
        lyrics: The known lyrics text.
        model_name: Whisper model to use.
        device: Device to use (None for auto-detect).

    Returns:
        dict with keys: "text", "language", "segments".
        Each segment contains "words" with "word", "start", "end".
    """
    if device is None:
        device = _detect_device()

    model = _get_model(model_name, device)

    filtered_lyrics = " ".join(lyrics.splitlines())

    print("Aligning lyrics with audio...")
    start = time.time()
    transcribe_kwargs: dict = {
        "word_timestamps": True,
        "initial_prompt": filtered_lyrics,
    }
    if device == "mps":
        transcribe_kwargs["fp16"] = False
    result: dict = model.transcribe(audio_path, **transcribe_kwargs)  # type: ignore
    elapsed = time.time() - start
    print(f"Alignment completed in {elapsed:.1f}s")

    return result
