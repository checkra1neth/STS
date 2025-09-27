"""Voice activity controller for streaming transcription."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class VACConfig:
    """Configuration for the voice activity controller."""

    samplerate: int
    chunk_duration: float
    overlap_ratio: float
    silence_threshold: float
    window_duration: float
    min_silence_duration: float
    speech_pad: float
    enabled: bool


class VoiceActivityController:
    """Aggregate audio frames and emit speech segments using optional VAD."""

    def __init__(self, config: VACConfig):
        self.config = config
        self._request_vac = config.enabled
        self._vad_model = None
        self._torch = None
        self._get_speech_timestamps = None
        self._load_error: Optional[str] = None

        if self._request_vac and config.samplerate not in (8000, 16000):
            self._load_error = "VAC поддерживает только 8 или 16 кГц"
            self._request_vac = False

        if self._request_vac:
            self._initialize_vad()

        self.using_vac = self._request_vac and self._vad_model is not None

        # Buffers for VAC path
        self._residual_mono = np.zeros(0, dtype=np.float32)
        self._residual_multi = np.zeros((0, 1), dtype=np.float32)
        self._vad_buffer_mono: List[np.ndarray] = []
        self._vad_buffer_multi: List[np.ndarray] = []
        self._vad_buffer_samples = 0

        # Buffers for RMS fallback path
        self._chunk_buffer: List[np.ndarray] = []
        self._chunk_samples = 0
        self._chunk_start_time: Optional[float] = None
        self._overlap_tail: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def push(self, frame: np.ndarray) -> List[np.ndarray]:
        """Consume raw audio frame and return speech-ready chunks."""

        if not frame.size:
            return []

        if self.using_vac:
            return self._push_with_vad(frame)

        return self._push_without_vad(frame)

    def finalize(self) -> List[np.ndarray]:
        """Flush pending data when recording stops."""

        if self.using_vac:
            return self._flush_vad(final=True)

        pending = []
        if self._chunk_buffer:
            combined = np.concatenate(self._chunk_buffer, axis=0)
            pending.append(combined)
            self._chunk_buffer = []
            self._chunk_samples = 0
            self._chunk_start_time = None

        self._overlap_tail = None

        return pending

    def should_skip_silence(self, chunk: np.ndarray) -> bool:
        """Return True if chunk should be dropped as silence in fallback mode."""

        if self.using_vac:
            return False

        if not chunk.size:
            return True

        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return rms < self.config.silence_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_vad(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            self._load_error = f"не удалось импортировать torch: {exc}"
            return

        try:
            model, utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            self._load_error = f"ошибка загрузки Silero VAD: {exc}"
            return

        # utils import torchaudio; if it's unavailable we gracefully fall back
        required = ["get_speech_timestamps"]
        for name in required:
            if name not in utils:
                self._load_error = "Silero utils не предоставили get_speech_timestamps"
                return

        self._vad_model = model
        self._torch = torch
        self._get_speech_timestamps = utils["get_speech_timestamps"]

        try:
            self._vad_model.to("cpu")
            self._vad_model.reset_states()
        except Exception:
            # Model might not expose reset_states for ONNX variant – ignore
            pass

    # --------------------------- VAC path -----------------------------
    def _push_with_vad(self, frame: np.ndarray) -> List[np.ndarray]:
        mono_frame, multi_frame = self._split_frame(frame)

        self._vad_buffer_mono.append(mono_frame)
        self._vad_buffer_multi.append(multi_frame)
        self._vad_buffer_samples += len(mono_frame)

        window_samples = max(int(self.config.window_duration * self.config.samplerate), 1)
        if self._vad_buffer_samples < window_samples:
            return []

        return self._flush_vad()

    def _flush_vad(self, final: bool = False) -> List[np.ndarray]:
        if not (self._vad_buffer_mono or self._residual_mono.size):
            return []

        mono_segments = []
        multi_segments = []

        if self._residual_mono.size:
            mono_segments.append(self._residual_mono)
            multi_segments.append(self._residual_multi)

        mono_segments.extend(self._vad_buffer_mono)
        multi_segments.extend(self._vad_buffer_multi)

        mono_window = np.concatenate(mono_segments, axis=0).astype(np.float32, copy=False)
        multi_window = np.concatenate(multi_segments, axis=0).astype(np.float32, copy=False)

        self._vad_buffer_mono = []
        self._vad_buffer_multi = []
        self._vad_buffer_samples = 0

        chunks = self._run_vad(mono_window, multi_window)

        if final:
            self._residual_mono = np.zeros(0, dtype=np.float32)
            self._residual_multi = np.zeros((0, multi_window.shape[1]), dtype=np.float32)
        else:
            keep_tail = max(int(self.config.speech_pad * self.config.samplerate), 0)
            tail_start = max(len(mono_window) - keep_tail, 0)
            self._residual_mono = mono_window[tail_start:]
            self._residual_multi = multi_window[tail_start:]

        return chunks

    def _run_vad(self, mono_window: np.ndarray, multi_window: np.ndarray) -> List[np.ndarray]:
        if not self._get_speech_timestamps or self._vad_model is None:
            return []

        tensor = self._torch.from_numpy(mono_window)
        try:
            timestamps = self._get_speech_timestamps(
                tensor,
                self._vad_model,
                threshold=0.5,
                sampling_rate=self.config.samplerate,
                min_speech_duration_ms=int(self.config.min_silence_duration * 1000),
                min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
                speech_pad_ms=int(self.config.speech_pad * 1000),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            self._load_error = f"ошибка работы Silero VAD: {exc}"
            self.using_vac = False
            return []

        if not timestamps:
            return []

        chunks: List[np.ndarray] = []
        channels = multi_window.shape[1]
        for ts in timestamps:
            start = max(int(ts.get("start", 0)), 0)
            end = max(int(ts.get("end", len(multi_window))), 0)
            end = min(end, len(multi_window))
            if end <= start:
                continue
            segment = multi_window[start:end]
            if channels == 1:
                chunks.append(segment.copy())
            else:
                chunks.append(segment.copy())

        return chunks

    # ----------------------- Fallback path ----------------------------
    def _push_without_vad(self, frame: np.ndarray) -> List[np.ndarray]:
        now = time.time()

        if self._overlap_tail is not None:
            self._chunk_buffer = [self._overlap_tail.copy()]
            self._chunk_samples = len(self._overlap_tail)
            self._overlap_tail = None
            self._chunk_start_time = self._chunk_start_time or now

        if not self._chunk_buffer:
            self._chunk_start_time = now

        self._chunk_buffer.append(frame)
        self._chunk_samples += len(frame)

        target_samples = max(int(self.config.chunk_duration * self.config.samplerate), self.config.samplerate // 2)
        duration_ready = False
        if self._chunk_start_time is not None:
            duration_ready = (now - self._chunk_start_time) >= self.config.chunk_duration

        if self._chunk_samples < target_samples and not duration_ready:
            return []

        combined = np.concatenate(self._chunk_buffer, axis=0)
        self._chunk_buffer = []
        self._chunk_samples = 0
        self._chunk_start_time = None

        if self.config.overlap_ratio > 0 and combined.size:
            overlap_samples = int(combined.shape[0] * self.config.overlap_ratio)
            if overlap_samples > 0:
                self._overlap_tail = combined[-overlap_samples:].copy()

        return [combined]

    # --------------------------- Utils --------------------------------
    def _split_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame.ndim == 1:
            mono = frame.astype(np.float32, copy=False)
            multi = mono[:, None]
            return mono, multi

        mono = frame.mean(axis=1).astype(np.float32, copy=False)
        multi = frame.astype(np.float32, copy=False)
        return mono, multi


__all__ = ["VoiceActivityController", "VACConfig"]
