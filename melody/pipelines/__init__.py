"""High-level pipelines for lyrics and karaoke generation."""

from melody.models import ctc_alignment, whisper_alignment


def normalize_alignment_result(
    result: dict, *, fallback_text: str | None, source: str
) -> dict:
    """Normalize alignment output to consistent schema.

    Args:
        result: Raw alignment/transcription result
        fallback_text: Text to use if result has no text
        source: "ctc" or "whisper"

    Returns:
        dict with keys: text, language, segments, source
    """
    text = result.get("text") or fallback_text or ""
    language = result.get("language")
    segments = []
    raw_segments = result.get("segments") or []

    if source == "ctc":
        words = raw_segments[0]["words"] if raw_segments else []

        # Split text into lines
        lines = text.split("\n")

        # Create a mapping of words to lines
        line_word_count = [len(line.split()) for line in lines]

        # Distribute words across lines
        word_idx = 0
        for line_idx, (line_text, word_count) in enumerate(zip(lines, line_word_count)):
            if word_count == 0:
                continue

            # Get words for this line
            line_words = words[word_idx : word_idx + word_count]
            word_idx += word_count

            if not line_words:
                continue

            segment_start = min((w["start"] for w in line_words), default=None)
            segment_end = max((w["end"] for w in line_words), default=None)

            segments.append(
                {
                    "id": line_idx,
                    "start": segment_start,
                    "end": segment_end,
                    "text": line_text.strip(),
                    "words": [
                        {
                            "word": w.get("word", "").strip(),
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "score": w.get("score"),
                            "singer": w.get("singer"),
                        }
                        for w in line_words
                    ],
                }
            )
    else:
        for idx, seg in enumerate(raw_segments):
            words = seg.get("words") or []
            segments.append(
                {
                    "id": seg.get("id", idx),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", "").strip(),
                    "words": [
                        {
                            "word": w.get("word", "").strip(),
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "score": w.get("probability"),
                            "singer": w.get("singer"),
                        }
                        for w in words
                    ],
                }
            )

    return {
        "text": text,
        "language": language,
        "segments": segments,
        "source": source,
    }


def process_lyrics(audio_path: str, lyrics: str | None = None) -> dict:
    """Process lyrics for audio file.

    Uses CTC forced alignment if lyrics provided (precise word-level timestamps).
    Uses Whisper transcription if no lyrics provided.

    Args:
        audio_path: Path to audio file (ideally isolated vocals)
        lyrics: Known lyrics text, or None to auto-transcribe

    Returns:
        dict with text, language, segments, and source
    """
    if lyrics is not None:
        result = ctc_alignment.align(audio_path, lyrics)
        normalized = normalize_alignment_result(
            result,
            fallback_text=lyrics,
            source="ctc",
        )
    else:
        result = whisper_alignment.transcribe_lyrics(audio_path)
        normalized = normalize_alignment_result(
            result,
            fallback_text=None,
            source="whisper",
        )

    return normalized
