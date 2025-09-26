import sys
import types
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


if __name__ == "__main__":  # pragma: no cover - test entry point
    unittest.main()
