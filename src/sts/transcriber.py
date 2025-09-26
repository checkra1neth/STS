"""Utility helpers for capturing audio and transcribing it with faster-whisper."""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class TranscriptionSegment:
    """A single segment returned by the Whisper model."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Aggregated transcription output for an audio file."""

    text: str
    segments: List[TranscriptionSegment]
    duration: float
    language: Optional[str]


def capture_audio(
    destination: Path,
    samplerate: int = 16_000,
    channels: int = 1,
    device: Optional[int | str] = None,
) -> Path:
    """Capture audio from the default input device until interrupted."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    print("Начинаю запись. Нажмите Ctrl+C, чтобы остановить захват аудио.")

    with sf.SoundFile(destination, mode="w", samplerate=samplerate, channels=channels) as file:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=callback,
            device=device,
            dtype="float32",
        ):
            try:
                while True:
                    file.write(audio_queue.get())
            except KeyboardInterrupt:
                print("Запись остановлена. Продолжаю транскрибацию...")

    return destination


def load_model(
    model_size: str = "distil-small",
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    """Load an optimized Whisper model via faster-whisper."""

    kwargs = {
        "device": device,
        "compute_type": compute_type,
    }

    if cpu_threads is not None:
        # The faster-whisper bindings expose thread configuration via intra/inter
        # parameters. Limiting the intra-threads count allows the caller to avoid
        # saturating the entire CPU on smaller machines.
        kwargs["intra_threads"] = cpu_threads

    return WhisperModel(
        model_size,
        **kwargs,
    )


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    *,
    beam_size: int = 5,
    language: Optional[str] = None,
    temperature: float = 0.0,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """Transcribe an audio file and aggregate the model output."""

    segment_iter, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
        temperature=temperature,
        vad_filter=vad_filter,
    )

    collected: List[TranscriptionSegment] = []
    text_parts: List[str] = []

    for segment in segment_iter:
        text = segment.text.strip()
        collected.append(
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=text,
            )
        )
        if text:
            text_parts.append(text)

    return TranscriptionResult(
        text=" ".join(text_parts),
        segments=collected,
        duration=getattr(info, "duration", 0.0),
        language=getattr(info, "language", None),
    )
