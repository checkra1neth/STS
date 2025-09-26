"""Speech-to-text utilities built around faster-whisper."""

from .transcriber import capture_audio, load_model, transcribe_audio

__all__ = [
    "capture_audio",
    "load_model",
    "transcribe_audio",
]
