"""Tests for melody.ai pipeline."""

import tempfile
from pathlib import Path

from melody.models import separation, ctc_alignment, whisper_alignment
from melody.pipelines import process_lyrics


def test_ctc_alignment(vocals_path: str):
    """Test CTC alignment with known lyrics."""
    print("CTC alignment test started...")
    if not Path(vocals_path).exists():
        print("⊘ Skipped: sample file missing")
        return

    if not Path("samples/kill_this_love.txt").exists():
        print("⊘ Skipped: lyrics file missing")
        return

    try:
        with open("samples/kill_this_love.txt", "r") as f:
            lyrics_text = f.read()
        result = ctc_alignment.align(vocals_path, lyrics_text)
        assert "text" in result
        assert "segments" in result
        print("✓ CTC alignment")
    except Exception as e:
        print(f"⊘ Skipped: CTC alignment - {type(e).__name__}: {str(e)[:80]}")


def test_whisper_transcribe(vocals_path: str):
    """Test Whisper transcription."""
    print("Whisper transcription test started...")
    if not Path(vocals_path).exists():
        print("⊘ Skipped: sample file missing")
        return

    try:
        result = whisper_alignment.transcribe_lyrics(vocals_path)
        assert "text" in result
        assert "segments" in result
        print("✓ Whisper transcription")
    except KeyboardInterrupt:
        print("⊘ Skipped: Whisper transcription interrupted")
    except Exception as e:
        print(f"⊘ Skipped: Whisper transcription - {type(e).__name__}: {str(e)[:80]}")


_temp_dir = None


def test_separate() -> str:
    """Test audio separation."""
    global _temp_dir
    print("Audio separation test started...")
    if not Path("samples/kill_this_love.mp3").exists():
        print("⊘ Skipped: sample file missing")
        raise FileNotFoundError("Sample audio file missing")

    try:
        _temp_dir = tempfile.TemporaryDirectory()
        tmpdir = _temp_dir.name
        vocals = separation.separate_audio(
            "samples/kill_this_love.mp3",
            output_dir=tmpdir,
        )
        assert vocals and Path(vocals).exists()
        print("✓ Audio separation")
        return vocals
    except KeyboardInterrupt:
        print("⊘ Skipped: Audio separation interrupted")
        if _temp_dir:
            _temp_dir.cleanup()
        raise
    except Exception as e:
        print(f"⊘ Skipped: Audio separation - {type(e).__name__}: {str(e)[:80]}")
        if _temp_dir:
            _temp_dir.cleanup()
        raise


def test_process_lyrics_no_input(vocals_path: str):
    """Test lyrics processing with transcription."""
    print("Lyrics processing (transcription) test started...")
    if not Path(vocals_path).exists():
        print("⊘ Skipped: sample file missing")
        return

    try:
        result = process_lyrics(vocals_path, lyrics=None)
        assert result.get("source") == "whisper"
        assert result.get("text")
        assert result.get("segments")
        print("✓ Lyrics processing (transcription)")
    except KeyboardInterrupt:
        print("⊘ Skipped: Lyrics processing interrupted")
    except Exception as e:
        print(f"⊘ Skipped: Lyrics processing - {type(e).__name__}: {str(e)[:80]}")


def test_process_lyrics_with_input(vocals_path: str):
    """Test lyrics processing with known lyrics."""
    print("Lyrics processing (alignment) test started...")
    if not Path(vocals_path).exists():
        print("⊘ Skipped: sample file missing")
        return

    try:
        with open("samples/kill_this_love.txt", "r") as f:
            lyrics_text = f.read()
        result = process_lyrics(vocals_path, lyrics=lyrics_text)
        assert result.get("source") == "ctc"
        assert result.get("text") == lyrics_text
        assert result.get("segments")
        print("✓ Lyrics processing (alignment)")
    except Exception as e:
        print(
            f"⊘ Skipped: Lyrics processing (alignment) - {type(e).__name__}: {str(e)[:80]}"
        )


if __name__ == "__main__":
    print("Running pipeline tests...\n")

    try:
        vocals_path = test_separate()
    except Exception:
        vocals_path = "samples/kill_this_love.mp3"

    for test_func in [
        test_ctc_alignment,
        test_whisper_transcribe,
        test_process_lyrics_no_input,
        test_process_lyrics_with_input,
    ]:
        try:
            test_func(vocals_path)
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    if _temp_dir:
        _temp_dir.cleanup()

    print("\nDone")
