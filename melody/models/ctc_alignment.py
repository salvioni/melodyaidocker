import re
import time
from melody.utils import detect_device
import torch
import torchaudio

# MMS_FA model constants
_SAMPLE_RATE = 16000
_FRAME_DURATION = 320 / _SAMPLE_RATE  # ~0.02s per frame

_model = None
_tokenizer = None
_aligner = None
_uroman = None

_resamplers = {}


def _get_uroman():
    global _uroman
    if _uroman is None:
        from uroman import Uroman

        _uroman = Uroman()
    return _uroman


def _get_mms_fa() -> tuple:
    global _model, _tokenizer, _aligner

    if _model is not None:
        return _model, _tokenizer, _aligner

    device = detect_device()
    bundle = torchaudio.pipelines.MMS_FA

    print(f"Loading MMS_FA model on {device}...")
    start = time.time()

    _model = bundle.get_model().to(device)
    _tokenizer = bundle.get_tokenizer()
    _aligner = bundle.get_aligner()

    elapsed = time.time() - start
    print(f"MMS_FA model loaded in {elapsed:.1f}s on {device}")

    return _model, _tokenizer, _aligner


def _romanize(text: str) -> str:
    """Romanize text to lowercase Latin characters + apostrophe."""
    uroman = _get_uroman()
    romanized: str = uroman.romanize_string(text)
    cleaned = re.sub(r"[^a-zA-Z' ]", "", romanized)
    return cleaned.lower()


def _prepare_words(lyrics: str) -> tuple[list[str], list[str]]:
    raw_words = lyrics.split()
    original_words = []
    romanized_words = []

    for word in raw_words:
        romanized = _romanize(word)
        if romanized.strip():
            original_words.append(word)
            romanized_words.append(romanized.strip())

    return original_words, romanized_words


def align(audio_path: str, lyrics: str) -> dict:
    """
    Align known lyrics to audio using CTC forced alignment (MMS_FA).
    """
    model, tokenizer, aligner = _get_mms_fa()
    device = next(model.parameters()).device

    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != _SAMPLE_RATE:
        if sr not in _resamplers:
            _resamplers[sr] = torchaudio.transforms.Resample(sr, _SAMPLE_RATE)
        waveform = _resamplers[sr](waveform)

    original_words, romanized_words = _prepare_words(lyrics)
    if not romanized_words:
        print("ERROR: no alignable words found in lyrics")
        return {"text": lyrics, "segments": []}

    token_sequences: list = tokenizer(romanized_words)

    print(f"Aligning {len(romanized_words)} words...")
    start = time.time()

    chunk_size = 30 * _SAMPLE_RATE
    emissions = []

    device_type = (
        "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
    )
    enable_autocast = device_type in ["mps", "cuda"]

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device_type, enabled=enable_autocast),
    ):
        for i in range(0, waveform.size(1), chunk_size):
            chunk = waveform[:, i : i + chunk_size].to(device)
            emission, _ = model(chunk)
            emissions.append(emission.cpu())

    final_emission = torch.cat(emissions, dim=1)

    aligned_spans: list = aligner(final_emission[0], token_sequences)

    elapsed = time.time() - start
    print(f"Alignment completed in {elapsed:.1f}s")

    words_result = []
    for i, (orig_word, spans) in enumerate(zip(original_words, aligned_spans)):
        if not spans:
            continue
        word_start = spans[0].start * _FRAME_DURATION
        word_end = spans[-1].end * _FRAME_DURATION
        word_score = sum(s.score for s in spans) / len(spans)
        words_result.append(
            {
                "word": orig_word,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "score": round(word_score, 3),
            }
        )

    return {
        "text": lyrics,
        "segments": [{"words": words_result}],
    }
