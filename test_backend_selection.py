import pytest

from melody.models import separation


def test_auto_backend_prefers_demucs_on_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MELODY_SEPARATION_BACKEND", raising=False)
    monkeypatch.setattr(separation, "detect_device", lambda: "cuda")

    assert separation.resolve_backend() == "demucs"


def test_auto_backend_keeps_uvr_off_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MELODY_SEPARATION_BACKEND", raising=False)
    monkeypatch.setattr(separation, "detect_device", lambda: "mps")

    assert separation.resolve_backend() == "uvr"


def test_backend_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MELODY_SEPARATION_BACKEND", "uvr")
    monkeypatch.setattr(separation, "detect_device", lambda: "cuda")

    assert separation.resolve_backend() == "uvr"


def test_invalid_backend_env_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MELODY_SEPARATION_BACKEND", "invalid")

    with pytest.raises(ValueError, match="MELODY_SEPARATION_BACKEND"):
        separation.resolve_backend()
