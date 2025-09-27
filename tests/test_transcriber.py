"""Lightweight tests for backend selection and simulation helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


# Provide lightweight stubs for optional heavy dependencies to keep the test
# environment minimal. Only the attributes referenced by ``transcriber`` during
# import are defined.
_install_stub_module(
    "numpy",
    ndarray=type("ndarray", (), {}),
    float32=float,
    array=lambda *args, **kwargs: None,
    max=max,
    abs=abs,
    concatenate=lambda *args, **kwargs: None,
    mean=lambda *args, **kwargs: 0.0,
)
_install_stub_module(
    "sounddevice",
    query_devices=lambda: [],
)
_install_stub_module(
    "soundfile",
    write=lambda *args, **kwargs: None,
)
_install_stub_module("ffmpeg")
_install_stub_module("faster_whisper", WhisperModel=object)

from sts import transcriber


class DummyModel:
    def __init__(self, model_size: str, *, compute_type: str, **kwargs):
        if compute_type == "int16":
            raise ValueError(
                "Requested int16 compute type, but the target device or backend do not support efficient int16 computation."
            )
        self.model_size = model_size
        self.compute_type = compute_type
        self.kwargs = kwargs
        self.is_multilingual = True

    def transcribe(self, *args, **kwargs):  # pragma: no cover - behaviour covered through wrapper
        segment = types.SimpleNamespace(start=0.0, end=1.0, text="hello")
        info = types.SimpleNamespace(duration=1.0, language="en")
        return [segment], info


class LoadModelFallbackTests(unittest.TestCase):
    def test_fallback_to_int8_when_int16_unavailable(self):
        with patch.object(transcriber, "WhisperModel", DummyModel):
            backend = transcriber.load_transcription_backend(
                "faster-whisper",
                "tiny",
                device="cpu",
                compute_type="auto",
            )

        self.assertIsInstance(backend.handle, DummyModel)
        self.assertEqual(backend.handle.compute_type, "int8")

    def test_transcribe_audio_delegates_to_backend(self):
        with patch.object(transcriber, "WhisperModel", DummyModel):
            backend = transcriber.load_transcription_backend(
                "faster-whisper",
                "tiny",
                device="cpu",
                compute_type="int8",
            )

        result = transcriber.transcribe_audio(backend, Path("dummy.wav"))
        self.assertEqual(result.text, "hello")
        self.assertEqual(len(result.segments), 1)


class ReplayTraceTests(unittest.TestCase):
    def test_replay_stream_trace_invokes_callback(self):
        events = [
            {"timestamp": 0.0, "text": "one", "language": "en"},
            {"timestamp": 0.5, "text": "two", "language": "en", "duration": 0.4},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            path = Path(handle.name)
            for event in events:
                handle.write(json.dumps(event) + "\n")

        received: list[dict[str, str]] = []

        try:
            transcriber.replay_stream_trace(
                path,
                callback=lambda payload: received.append(payload),
                speed=50.0,  # ускоряем во избежание задержек
            )
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["text"], "one")
        self.assertEqual(received[1]["text"], "two")


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()

