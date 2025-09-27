"""Speech-to-text utilities built around faster-whisper."""

from .transcriber import (
    capture_audio,
    load_model,
    load_transcription_backend,
    transcribe_audio,
    extract_audio_from_video,
    is_video_file,
    capture_system_audio,
    find_system_audio_devices,
    stream_transcribe,
    StreamTranscriber,
    get_best_stream_profile,
    list_backend_names,
    DEFAULT_BACKEND_NAME,
    replay_stream_trace,
    load_stream_trace,
)

__all__ = [
    "capture_audio",
    "load_model",
    "load_transcription_backend",
    "transcribe_audio",
    "extract_audio_from_video",
    "is_video_file",
    "capture_system_audio",
    "find_system_audio_devices",
    "stream_transcribe",
    "StreamTranscriber",
    "get_best_stream_profile",
    "list_backend_names",
    "DEFAULT_BACKEND_NAME",
    "replay_stream_trace",
    "load_stream_trace",
]
