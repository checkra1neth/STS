import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch


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
    max=lambda *args, **kwargs: 0,
    abs=abs,
    concatenate=lambda *args, **kwargs: None,
    mean=lambda *args, **kwargs: 0.0,
)
_install_stub_module("sounddevice")
_install_stub_module("soundfile")
_install_stub_module("ffmpeg")
_install_stub_module("faster_whisper", WhisperModel=object)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sts import transcriber


class DummyModel:
    def __init__(self, model_size: str, *, compute_type: str, **kwargs):
        # Simulate backend rejecting int16 compute type
        if compute_type == "int16":
            raise ValueError(
                "Requested int16 compute type, but the target device or backend do not support efficient int16 computation."
            )
        self.model_size = model_size
        self.compute_type = compute_type
        self.kwargs = kwargs


class LoadModelFallbackTests(unittest.TestCase):
    def test_fallback_to_int8_when_int16_unavailable(self):
        with patch.object(transcriber, "WhisperModel", DummyModel):
            model = transcriber.load_model("tiny", device="cpu", compute_type="auto")

        self.assertIsInstance(model, DummyModel)
        self.assertEqual(model.compute_type, "int8")


class BufferManagerTests(unittest.TestCase):
    def test_sentence_manager_emits_after_punctuation(self):
        manager = transcriber.SentenceBufferManager(pause_threshold=0.5, trimming_enabled=False)

        emitted = manager.push(
            [
                transcriber.TranscriptionSegment(start=0.0, end=0.4, text="Привет,"),
                transcriber.TranscriptionSegment(start=0.4, end=0.7, text="мир."),
            ],
            chunk_offset=0.0,
            chunk_duration=0.7,
        )

        self.assertEqual(emitted, ["Привет, мир."])

    def test_sentence_manager_uses_pause_to_finalize(self):
        manager = transcriber.SentenceBufferManager(pause_threshold=0.6, trimming_enabled=False)

        first = manager.push(
            [transcriber.TranscriptionSegment(start=0.0, end=0.5, text="Сегодня хорошая")],
            chunk_offset=0.0,
            chunk_duration=0.5,
        )
        self.assertEqual(first, [])

        second = manager.push(
            [transcriber.TranscriptionSegment(start=0.0, end=0.4, text="погода")],
            chunk_offset=1.4,
            chunk_duration=0.4,
        )

        self.assertEqual(second, ["Сегодня хорошая"])

    def test_trimming_flushes_stale_buffer(self):
        manager = transcriber.SentenceBufferManager(
            pause_threshold=0.5,
            trimming_enabled=True,
            trimming_window=0.5,
        )

        manager.push(
            [transcriber.TranscriptionSegment(start=0.0, end=0.2, text="Незавершённая фраза")],
            chunk_offset=0.0,
            chunk_duration=0.2,
        )

        flushed = manager.push(
            [],
            chunk_offset=2.0,
            chunk_duration=0.0,
        )

        self.assertEqual(flushed, ["Незавершённая фраза"])


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()
