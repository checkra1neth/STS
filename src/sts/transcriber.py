"""Utility helpers for capturing audio and transcribing it with faster-whisper."""

from __future__ import annotations

import math
from collections import deque
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Union

import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False


@dataclass
class StreamProfile:
    """Tuning parameters for low-latency streaming transcription."""

    name: str
    chunk_duration: float
    overlap_ratio: float
    beam_size: int
    vad_filter: bool
    temperature: float
    use_threading: bool
    cpu_threads: Optional[int]
    buffer_strategy: str = "sentence"
    buffer_pause: float = 0.6
    buffer_trimming: bool = True
    buffer_trimming_sec: float = 12.0


BEST_STREAM_PROFILE = StreamProfile(
    name="balanced_realtime",
    chunk_duration=2.4,
    overlap_ratio=0.16,
    beam_size=5,
    vad_filter=True,
    temperature=0.0,
    use_threading=True,
    cpu_threads=4,
    buffer_strategy="sentence",
    buffer_pause=0.6,
    buffer_trimming=True,
    buffer_trimming_sec=12.0,
)


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


def _calculate_overlap(previous: str, current: str, *, max_overlap_chars: int = 160) -> int:
    """Return the number of overlapping characters between the tail of previous and head of current."""

    if not previous or not current:
        return 0

    previous_tail = previous[-max_overlap_chars :].lower()
    current_lower = current.lower()
    max_len = min(len(previous_tail), len(current_lower))

    for overlap in range(max_len, 0, -1):
        if previous_tail[-overlap:] == current_lower[:overlap]:
            return overlap

    return 0


def _looks_like_duplicate(candidate: str, previous: str) -> bool:
    """Heuristically determine if candidate repeats previous content."""

    if not candidate or not previous:
        return False

    candidate_lower = candidate.lower()
    previous_lower = previous.lower()

    if candidate_lower == previous_lower:
        return True

    shorter, longer = (
        (candidate_lower, previous_lower)
        if len(candidate_lower) <= len(previous_lower)
        else (previous_lower, candidate_lower)
    )

    if len(shorter) >= 12 and shorter in longer:
        return True

    overlap = _calculate_overlap(previous_lower, candidate_lower, max_overlap_chars=len(candidate_lower))
    if overlap >= max(10, int(len(candidate_lower) * 0.75)):
        return True

    return False


def _strip_repeated_sequences(text: str) -> str:
    """Remove obvious repeated words or short phrases from text."""

    if not text:
        return text

    words = text.split()
    if len(words) < 2:
        return text.strip()

    cleaned: List[str] = []
    i = 0

    while i < len(words):
        remaining = len(words) - i
        max_window = min(6, remaining // 2)
        window_found = 0
        reference_lower: Optional[List[str]] = None

        for window in range(max_window, 0, -1):
            first = words[i : i + window]
            second = words[i + window : i + 2 * window]
            if not second:
                continue
            if all(f.lower() == s.lower() for f, s in zip(first, second)):
                window_found = window
                reference_lower = [w.lower() for w in first]
                break

        if window_found:
            cleaned.extend(words[i : i + window_found])
            i += window_found

            if reference_lower is None:
                reference_lower = [w.lower() for w in words[i - window_found : i]]

            # Пропускаем повторяющиеся блоки той же длины.
            while i + window_found <= len(words):
                candidate = words[i : i + window_found]
                if all(c.lower() == ref for c, ref in zip(candidate, reference_lower)):
                    i += window_found
                else:
                    break
        else:
            cleaned.append(words[i])
            i += 1

    normalized = " ".join(cleaned)
    return normalized.strip()


def _assemble_transcription(parts: Sequence[str]) -> str:
    """Join cleaned transcription parts while avoiding repeated fragments."""

    collected: List[str] = []
    cumulative = ""

    for part in parts:
        cleaned = _strip_repeated_sequences(part.strip())
        if not cleaned:
            continue

        overlap = _calculate_overlap(cumulative, cleaned)
        if overlap:
            cleaned = cleaned[overlap:].lstrip()

        if cleaned and any(_looks_like_duplicate(cleaned, previous) for previous in collected[-3:]):
            continue

        if cleaned:
            collected.append(cleaned)
            cumulative = f"{cumulative} {cleaned}".strip()

    return " ".join(collected)


def _prepare_audio_array(audio_data: np.ndarray) -> np.ndarray:
    """Convert arbitrary audio chunk to mono float32 array suitable for inference."""

    if audio_data.ndim > 1:
        # Усредняем каналы для повышения устойчивости и избегания повторов
        audio_data = audio_data.mean(axis=1)

    mono = audio_data.astype(np.float32, copy=False)

    # Ограничиваем амплитуду, если входящий сигнал оказывается слишком громким
    peak = np.max(np.abs(mono)) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak

    return mono


def _deduplicate_candidate(text: str, history: Deque[str]) -> str:
    """Clean text and drop it if it heavily overlaps with previously emitted fragments."""

    cleaned = _strip_repeated_sequences(text.strip())
    if not cleaned:
        return ""

    previous = " ".join(history)
    overlap = _calculate_overlap(previous, cleaned)
    if overlap:
        cleaned = cleaned[overlap:].lstrip()

    for recent in reversed(history):
        if _looks_like_duplicate(cleaned, recent):
            return ""

    return cleaned


@dataclass
class _BufferedSegment:
    start: float
    end: float
    text: str


class BaseBufferManager:
    """Base class for buffering strategies that smoothen streaming transcripts."""

    def __init__(
        self,
        *,
        history_size: int = 12,
        trimming_enabled: bool = True,
        trimming_window: float = 12.0,
    ) -> None:
        self._history: Deque[str] = deque(maxlen=history_size)
        self._trimming_enabled = trimming_enabled
        self._trimming_window = trimming_window

    def reset(self) -> None:
        self._history.clear()

    def configure_trimming(self, enabled: Optional[bool] = None, window: Optional[float] = None) -> None:
        if enabled is not None:
            self._trimming_enabled = enabled
        if window is not None:
            self._trimming_window = max(window, 0.0)

    @property
    def trimming_enabled(self) -> bool:
        return self._trimming_enabled

    @property
    def trimming_window(self) -> float:
        return self._trimming_window

    def _append_history(self, text: str) -> str:
        deduplicated = _deduplicate_candidate(text, self._history)
        if deduplicated:
            self._history.append(deduplicated)
        return deduplicated

    def push(
        self,
        segments: Sequence[TranscriptionSegment],
        *,
        chunk_offset: float,
        chunk_duration: float,
    ) -> List[str]:
        raise NotImplementedError

    def finalize(self) -> List[str]:
        return []


class SegmentBufferManager(BaseBufferManager):
    """Emit each Whisper segment individually with duplicate suppression."""

    def push(
        self,
        segments: Sequence[TranscriptionSegment],
        *,
        chunk_offset: float,
        chunk_duration: float,
    ) -> List[str]:
        emitted: List[str] = []
        for segment in segments:
            cleaned = self._append_history(segment.text)
            if cleaned:
                emitted.append(cleaned)
        return emitted


class SentenceBufferManager(BaseBufferManager):
    """Accumulate segments until a pause or punctuation indicates sentence stability."""

    def __init__(
        self,
        *,
        pause_threshold: float = 0.6,
        sentence_endings: str = ".!?…",
        history_size: int = 12,
        trimming_enabled: bool = True,
        trimming_window: float = 12.0,
    ) -> None:
        super().__init__(
            history_size=history_size,
            trimming_enabled=trimming_enabled,
            trimming_window=trimming_window,
        )
        self._pause_threshold = max(pause_threshold, 0.0)
        self._sentence_endings = sentence_endings
        self._buffer: List[_BufferedSegment] = []

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()

    def push(
        self,
        segments: Sequence[TranscriptionSegment],
        *,
        chunk_offset: float,
        chunk_duration: float,
    ) -> List[str]:
        emitted: List[str] = []

        for segment in segments:
            cleaned_text = _strip_repeated_sequences(segment.text.strip())
            if not cleaned_text:
                continue

            buffered = _BufferedSegment(
                start=chunk_offset + segment.start,
                end=chunk_offset + segment.end,
                text=cleaned_text,
            )
            self._buffer.append(buffered)

        emitted.extend(self._finalize_ready_segments())

        if self.trimming_enabled:
            emitted.extend(self._trim_stale_segments(chunk_offset + chunk_duration))

        return emitted

    def _finalize_ready_segments(self) -> List[str]:
        emitted: List[str] = []

        while True:
            finalize_index = self._find_finalizable_index()
            if finalize_index is None:
                break

            chunk_text = " ".join(segment.text for segment in self._buffer[: finalize_index + 1])
            del self._buffer[: finalize_index + 1]

            confirmed = self._append_history(chunk_text)
            if confirmed:
                emitted.append(confirmed)

        return emitted

    def _find_finalizable_index(self) -> Optional[int]:
        finalize_index: Optional[int] = None

        for idx, segment in enumerate(self._buffer):
            stripped = segment.text.rstrip()
            ends_sentence = bool(stripped) and stripped[-1] in self._sentence_endings

            long_pause = False
            if idx + 1 < len(self._buffer):
                next_segment = self._buffer[idx + 1]
                long_pause = (next_segment.start - segment.end) >= self._pause_threshold

            if ends_sentence or long_pause:
                finalize_index = idx

        return finalize_index

    def _trim_stale_segments(self, current_time: float) -> List[str]:
        if not self._buffer:
            return []

        threshold = current_time - self.trimming_window
        if threshold <= 0:
            return []

        finalize_index = -1
        for idx, segment in enumerate(self._buffer):
            if segment.end <= threshold:
                finalize_index = idx
            else:
                break

        if finalize_index < 0:
            return []

        chunk_text = " ".join(segment.text for segment in self._buffer[: finalize_index + 1])
        del self._buffer[: finalize_index + 1]

        confirmed = self._append_history(chunk_text)
        return [confirmed] if confirmed else []

    def finalize(self) -> List[str]:
        if not self._buffer:
            return []

        chunk_text = " ".join(segment.text for segment in self._buffer)
        self._buffer.clear()

        confirmed = self._append_history(chunk_text)
        return [confirmed] if confirmed else []


def create_buffer_manager(
    strategy: str,
    *,
    pause_threshold: float = 0.6,
    trimming_enabled: bool = True,
    trimming_window: float = 12.0,
) -> BaseBufferManager:
    normalized = strategy.lower()
    if normalized == "segment":
        manager = SegmentBufferManager(
            trimming_enabled=trimming_enabled,
            trimming_window=trimming_window,
        )
    else:
        manager = SentenceBufferManager(
            pause_threshold=pause_threshold,
            trimming_enabled=trimming_enabled,
            trimming_window=trimming_window,
        )

    return manager


def get_best_stream_profile() -> StreamProfile:
    """Return the tuned preset that balances quality and latency."""

    return BEST_STREAM_PROFILE


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


def _recommend_compute_type(device: str) -> str:
    """Pick the best compute type for a given execution device."""

    normalized = (device or "cpu").lower()

    if normalized.startswith("cuda"):
        return "float16"

    if normalized.startswith("metal"):
        return "float16"

    if normalized.startswith("cpu"):
        return "int16"

    return "int8"


def load_model(
    model_size: str = "small",
    *,
    device: str = "cpu",
    compute_type: str = "auto",
    cpu_threads: Optional[int] = None,
) -> WhisperModel:
    """Load an optimized Whisper model via faster-whisper.

    When compute_type is set to ``auto`` the function chooses a high-quality
    configuration tailored to the selected device. CPU inference uses ``int16``
    activations for noticeably better accuracy compared to pure int8
    quantization, while GPU/Metal backends default to fast ``float16``
    execution.
    """

    if compute_type == "auto":
        recommended = _recommend_compute_type(device)
        candidate_types: List[str] = [recommended]
        if recommended != "int8":
            candidate_types.append("int8")
    else:
        candidate_types = [compute_type]
        if compute_type == "int16":
            candidate_types.append("int8")

    # Preserve order but avoid duplicate attempts if the fallback matches the
    # primary configuration.
    seen: Set[str] = set()
    unique_candidates = []
    for candidate in candidate_types:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    kwargs = {
        "device": device,
    }

    if cpu_threads is not None:
        # The faster-whisper bindings expose thread configuration via num_workers
        # parameters. Limiting the thread count allows the caller to avoid
        # saturating the entire CPU on smaller machines.
        kwargs["num_workers"] = cpu_threads

    last_error: Optional[Exception] = None
    for candidate in unique_candidates:
        try:
            return WhisperModel(
                model_size,
                compute_type=candidate,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on backend availability
            last_error = exc
            error_message = str(exc).lower()
            is_int16_candidate = "int16" in candidate
            mentions_int16_failure = "int16" in error_message
            is_int16_failure = is_int16_candidate and mentions_int16_failure

            # Fall back only when the backend explicitly complains about int16
            # support. Otherwise re-raise to avoid hiding real issues.
            if not is_int16_failure or candidate == unique_candidates[-1]:
                raise

            print(
                "⚠️  compute_type='int16' недоступен на данном устройстве. "
                "Пробую fallback на compute_type='int8'.",
                file=sys.stderr,
            )
            continue

    # Exhausted all candidates without success.
    if last_error is not None:
        raise last_error

    # This point should be unreachable because the loop either returns a model
    # or raises an exception. Raising a RuntimeError provides a clear message if
    # the invariant is ever broken by future refactoring.
    raise RuntimeError("Не удалось загрузить модель Whisper: отсутствуют варианты compute_type")


def transcribe_audio(
    model: WhisperModel,
    audio_source: Union[Path, np.ndarray],
    *,
    beam_size: int = 5,
    language: Optional[str] = None,
    temperature: float = 0.0,
    vad_filter: bool = True,
    compression_ratio_threshold: float = 2.0,
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = True,
) -> TranscriptionResult:
    """Transcribe an audio file and aggregate the model output."""

    if isinstance(audio_source, Path):
        audio_input: Union[str, np.ndarray] = str(audio_source)
    else:
        audio_input = audio_source

    segment_iter, info = model.transcribe(
        audio_input,
        beam_size=beam_size,
        language=language,
        temperature=temperature,
        vad_filter=vad_filter,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
    )

    collected: List[TranscriptionSegment] = []
    text_parts: List[str] = []
    cumulative = ""

    for segment in segment_iter:
        text = _strip_repeated_sequences(segment.text.strip())
        if text:
            overlap = _calculate_overlap(cumulative, text)
            if overlap:
                text = text[overlap:].lstrip()

        collected.append(
            TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=text,
            )
        )
        if text:
            text_parts.append(text)
            cumulative = f"{cumulative} {text}".strip()

    aggregated_text = _assemble_transcription(text_parts)

    return TranscriptionResult(
        text=aggregated_text,
        segments=collected,
        duration=getattr(info, "duration", 0.0),
        language=getattr(info, "language", None),
    )


def extract_audio_from_video(
    video_path: Path,
    output_path: Optional[Path] = None,
    samplerate: int = 16_000,
) -> Path:
    """Extract audio from video file using ffmpeg."""
    
    if not HAS_FFMPEG:
        raise ImportError(
            "ffmpeg-python не установлен. Установите его командой: pip install ffmpeg-python"
        )
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")
    
    if output_path is None:
        output_path = video_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Извлекаю аудио из {video_path.name}...")
    
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',
                ar=samplerate,
                ac=1,  # моно
                loglevel='error'
            )
            .overwrite_output()
            .run()
        )
        print(f"Аудио сохранено в {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        raise RuntimeError(f"Ошибка при извлечении аудио: {e}")


def is_video_file(file_path: Path) -> bool:
    """Check if file is a video file based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return file_path.suffix.lower() in video_extensions


def find_system_audio_devices() -> List[dict]:
    """Find potential system audio devices (like BlackHole, Loopback, etc.)."""
    devices = sd.query_devices()
    system_audio_devices = []
    
    # Ключевые слова для поиска системных аудиоустройств
    system_keywords = [
        'blackhole', 'loopback', 'soundflower', 'virtual', 'system', 
        'aggregate', 'multi-output', 'stereo mix', 'what u hear'
    ]
    
    for i, device in enumerate(devices):
        device_name = device['name'].lower()
        if any(keyword in device_name for keyword in system_keywords):
            if device['max_input_channels'] > 0:  # Устройство может записывать
                system_audio_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'samplerate': device['default_samplerate']
                })
    
    return system_audio_devices


def capture_system_audio(
    destination: Path,
    samplerate: int = 16_000,
    channels: int = 2,  # Системное аудио обычно стерео
    device: Optional[int | str] = None,
    duration: Optional[float] = None,
) -> Path:
    """Capture system audio (from streams, browser, etc.) until interrupted or duration reached."""
    
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Если устройство не указано, попробуем найти системное аудиоустройство
    if device is None:
        system_devices = find_system_audio_devices()
        if system_devices:
            device = system_devices[0]['index']
            print(f"Используется системное аудиоустройство: {system_devices[0]['name']}")
        else:
            print("Системное аудиоустройство не найдено. Используется устройство по умолчанию.")
            print("Для захвата системного аудио на macOS установите BlackHole:")
            print("brew install blackhole-2ch")

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
    recording = True

    def callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        if recording:
            audio_queue.put(indata.copy())

    if duration:
        print(f"Начинаю запись системного аудио на {duration} секунд...")
    else:
        print("Начинаю запись системного аудио. Нажмите Ctrl+C для остановки.")

    with sf.SoundFile(destination, mode="w", samplerate=samplerate, channels=channels) as file:
        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=callback,
            device=device,
            dtype="float32",
        ):
            try:
                if duration:
                    # Записываем определенное время
                    import time
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        if not audio_queue.empty():
                            file.write(audio_queue.get())
                        else:
                            time.sleep(0.01)
                    recording = False
                    print("Запись завершена по времени.")
                else:
                    # Записываем до прерывания
                    while True:
                        file.write(audio_queue.get())
            except KeyboardInterrupt:
                recording = False
                print("Запись остановлена. Продолжаю транскрибацию...")

    return destination


class StreamTranscriber:
    """Class for managing streaming transcription with proper stop control."""

    def __init__(self):
        self.recording = False
        self.audio_stream = None
        self.processing_threads = []  # Множественные потоки
        self.audio_queue = None
        self.result_queue = None
        self.max_threads = 4  # Максимум параллельных потоков обработки
        self.silence_threshold = 0.0015
        self._cumulative_text = ""
        self._buffer_manager: Optional[BaseBufferManager] = None
        self._chunk_lock = threading.Lock()
        self._pending_chunks: Dict[int, tuple[float, float, float, TranscriptionResult]] = {}
        self._next_chunk_index = 0
        self._next_chunk_to_emit = 0
        self._timeline_cursor = 0.0
        self._callback = None
        self._last_language: Optional[str] = None

    def start(
        self,
        model: WhisperModel,
        device: Optional[int | str] = None,
        samplerate: int = 16_000,
        channels: int = 2,
        chunk_duration: float = 5.0,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        overlap_ratio: float = 0.25,
        temperature: float = 0.0,
        use_threading: bool = True,
        buffer_strategy: str = "sentence",
        buffer_pause: float = 0.6,
        buffer_trimming: bool = True,
        buffer_trimming_sec: float = 12.0,
        callback=None,
    ):
        """Start streaming transcription."""

        if self.recording:
            return False
            
        # Если устройство не указано, попробуем найти системное аудиоустройство
        if device is None:
            system_devices = find_system_audio_devices()
            if system_devices:
                device = system_devices[0]['index']
                print(f"Используется системное аудиоустройство: {system_devices[0]['name']}")
            else:
                print("Системное аудиоустройство не найдено.")
        
        self.audio_queue = queue.Queue(maxsize=100)  # Большая очередь
        self.result_queue = queue.Queue()
        self.recording = True
        self._cumulative_text = ""
        self._pending_chunks.clear()
        self._next_chunk_index = 0
        self._next_chunk_to_emit = 0
        self._timeline_cursor = 0.0
        self._callback = callback
        self._last_language = None

        self._buffer_manager = create_buffer_manager(
            buffer_strategy,
            pause_threshold=buffer_pause,
            trimming_enabled=buffer_trimming,
            trimming_window=buffer_trimming_sec,
        )
        self._buffer_manager.reset()

        # Создаем пул потоков для обработки
        import concurrent.futures
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads)
        
        def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
            if status:
                print(status, file=sys.stderr)
            if self.recording:
                self.audio_queue.put(indata.copy())
        
        min_chunk_duration = max(chunk_duration, 0.4)
        target_chunk_samples = max(int(min_chunk_duration * samplerate), samplerate // 2)

        def process_audio_chunks():
            """Process audio chunks in separate thread."""
            chunk_data: List[np.ndarray] = []
            chunk_start_time = time.time()
            chunk_samples = 0

            while self.recording or not self.audio_queue.empty():
                try:
                    # Получаем аудиоданные с таймаутом
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    chunk_data.append(audio_chunk)
                    chunk_samples += len(audio_chunk)

                    # Проверяем, прошло ли достаточно времени для обработки чанка
                    current_time = time.time()
                    duration_ready = current_time - chunk_start_time >= chunk_duration
                    samples_ready = chunk_samples >= target_chunk_samples

                    if (duration_ready or samples_ready) and chunk_data:
                        combined_chunk = np.concatenate(chunk_data, axis=0)
                        processed_samples = combined_chunk.shape[0]

                        overlap_samples = 0

                        if processed_samples and self.recording:
                            chunk_duration_sec = processed_samples / float(samplerate) if samplerate else 0.0
                            overlap_samples = int(processed_samples * overlap_ratio) if overlap_ratio > 0 else 0
                            non_overlap_samples = max(processed_samples - overlap_samples, 0)

                            chunk_offset = self._timeline_cursor
                            self._timeline_cursor += non_overlap_samples / float(samplerate) if samplerate else 0.0

                            chunk_index = self._next_chunk_index
                            self._next_chunk_index += 1

                            chunk_rms = math.sqrt(float(np.mean(combined_chunk**2))) if combined_chunk.size else 0.0

                            if chunk_rms < self.silence_threshold:
                                silent_result = TranscriptionResult(
                                    text="",
                                    segments=[],
                                    duration=chunk_duration_sec,
                                    language=None,
                                )
                                self._register_chunk_result(
                                    chunk_index,
                                    chunk_offset,
                                    chunk_duration_sec,
                                    current_time,
                                    silent_result,
                                    callback,
                                )
                            else:
                                def process_chunk_instant(
                                    chunk_array=combined_chunk,
                                    emitted_time=current_time,
                                    local_chunk_index=chunk_index,
                                    local_chunk_offset=chunk_offset,
                                    local_chunk_duration=chunk_duration_sec,
                                ):
                                    empty_result = TranscriptionResult(
                                        text="",
                                        segments=[],
                                        duration=local_chunk_duration,
                                        language=None,
                                    )

                                    try:
                                        if not chunk_array.size:
                                            self._register_chunk_result(
                                                local_chunk_index,
                                                local_chunk_offset,
                                                local_chunk_duration,
                                                emitted_time,
                                                empty_result,
                                                callback,
                                            )
                                            return

                                        audio_array = _prepare_audio_array(chunk_array)
                                        if not audio_array.size:
                                            self._register_chunk_result(
                                                local_chunk_index,
                                                local_chunk_offset,
                                                local_chunk_duration,
                                                emitted_time,
                                                empty_result,
                                                callback,
                                            )
                                            return

                                        result = transcribe_audio(
                                            model,
                                            audio_array,
                                            language=language,
                                            beam_size=beam_size,
                                            temperature=temperature,
                                            vad_filter=vad_filter,
                                            compression_ratio_threshold=1.8,
                                            condition_on_previous_text=False,
                                        )

                                    except Exception as e:
                                        if self.recording:
                                            print(f"Ошибка: {e}")

                                        self._register_chunk_result(
                                            local_chunk_index,
                                            local_chunk_offset,
                                            local_chunk_duration,
                                            emitted_time,
                                            empty_result,
                                            callback,
                                        )
                                    else:
                                        self._register_chunk_result(
                                            local_chunk_index,
                                            local_chunk_offset,
                                            local_chunk_duration,
                                            emitted_time,
                                            result,
                                            callback,
                                        )

                                if use_threading:
                                    self.thread_pool.submit(process_chunk_instant)
                                else:
                                    process_chunk_instant()

                        if overlap_ratio > 0 and processed_samples > 0:
                            if overlap_samples > 0:
                                chunk_data = [combined_chunk[-overlap_samples:]]
                                chunk_samples = overlap_samples
                            else:
                                chunk_data = []
                                chunk_samples = 0
                        else:
                            chunk_data = []
                            chunk_samples = 0

                        chunk_start_time = current_time

                except queue.Empty:
                    continue
                except Exception as e:
                    if self.recording:
                        print(f"Ошибка обработки аудио: {e}")
                    break

            if chunk_data and chunk_samples:
                combined_chunk = np.concatenate(chunk_data, axis=0)

                def flush_chunk(chunk_array=combined_chunk):
                    chunk_duration_sec = chunk_array.shape[0] / float(samplerate) if samplerate else 0.0
                    chunk_offset = self._timeline_cursor
                    self._timeline_cursor += chunk_duration_sec

                    chunk_index = self._next_chunk_index
                    self._next_chunk_index += 1

                    timestamp = time.time()
                    empty_result = TranscriptionResult(
                        text="",
                        segments=[],
                        duration=chunk_duration_sec,
                        language=None,
                    )

                    try:
                        if not chunk_array.size:
                            self._register_chunk_result(
                                chunk_index,
                                chunk_offset,
                                chunk_duration_sec,
                                timestamp,
                                empty_result,
                                callback,
                            )
                            return

                        rms = math.sqrt(float(np.mean(chunk_array**2))) if chunk_array.size else 0.0

                        if rms < self.silence_threshold:
                            silent_result = TranscriptionResult(
                                text="",
                                segments=[],
                                duration=chunk_duration_sec,
                                language=None,
                            )
                            self._register_chunk_result(
                                chunk_index,
                                chunk_offset,
                                chunk_duration_sec,
                                timestamp,
                                silent_result,
                                callback,
                            )
                            return

                        audio_array = _prepare_audio_array(chunk_array)
                        if not audio_array.size:
                            self._register_chunk_result(
                                chunk_index,
                                chunk_offset,
                                chunk_duration_sec,
                                timestamp,
                                empty_result,
                                callback,
                            )
                            return

                        result = transcribe_audio(
                            model,
                            audio_array,
                            language=language,
                            beam_size=beam_size,
                            temperature=temperature,
                            vad_filter=vad_filter,
                            compression_ratio_threshold=1.8,
                            condition_on_previous_text=False,
                        )

                    except Exception as e:
                        print(f"Ошибка: {e}")
                        self._register_chunk_result(
                            chunk_index,
                            chunk_offset,
                            chunk_duration_sec,
                            timestamp,
                            empty_result,
                            callback,
                        )
                    else:
                        self._register_chunk_result(
                            chunk_index,
                            chunk_offset,
                            chunk_duration_sec,
                            timestamp,
                            result,
                            callback,
                        )

                if use_threading:
                    self.thread_pool.submit(flush_chunk)
                else:
                    flush_chunk()
        
        print("Начинаю потоковую транскрибацию.")

        # Запускаем обработку аудио в отдельном потоке
        self.processing_thread = threading.Thread(target=process_audio_chunks)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.processing_threads = [self.processing_thread]
        
        try:
            self.audio_stream = sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                callback=audio_callback,
                device=device,
                dtype="float32",
            )
            self.audio_stream.start()
            return True
            
        except Exception as e:
            print(f"Ошибка запуска аудиопотока: {e}")
            self.recording = False
            return False

    def _register_chunk_result(
        self,
        chunk_index: int,
        chunk_offset: float,
        chunk_duration: float,
        emitted_time: float,
        result: TranscriptionResult,
        callback,
    ) -> None:
        if self._buffer_manager is None:
            return

        with self._chunk_lock:
            self._pending_chunks[chunk_index] = (chunk_offset, chunk_duration, emitted_time, result)

            while self._next_chunk_to_emit in self._pending_chunks:
                (ready_offset, ready_duration, ready_time, ready_result) = self._pending_chunks.pop(self._next_chunk_to_emit)
                self._next_chunk_to_emit += 1

                additions = self._buffer_manager.push(
                    ready_result.segments,
                    chunk_offset=ready_offset,
                    chunk_duration=ready_duration,
                )

                for addition in additions:
                    if not addition:
                        continue

                    self._cumulative_text = f"{self._cumulative_text} {addition}".strip()

                    language = ready_result.language or self._last_language
                    if ready_result.language:
                        self._last_language = ready_result.language

                    effective_callback = callback or self._callback
                    if effective_callback:
                        effective_callback({
                            'text': addition,
                            'language': language,
                            'timestamp': ready_time,
                            'duration': ready_result.duration,
                        })

    def stop(self):
        """Stop streaming transcription."""
        if not self.recording:
            return
            
        print("Останавливаю потоковую транскрибацию...")
        self.recording = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Останавливаем пул потоков
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

        # Останавливаем основной поток обработки
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=0.5)

        final_chunks: List[str] = []
        if self._buffer_manager is not None:
            final_chunks = self._buffer_manager.finalize()

        for addition in final_chunks:
            if not addition:
                continue

            self._cumulative_text = f"{self._cumulative_text} {addition}".strip()

            effective_callback = self._callback
            if effective_callback:
                effective_callback({
                    'text': addition,
                    'language': self._last_language,
                    'timestamp': time.time(),
                    'duration': 0.0,
                })

        self._pending_chunks.clear()

        print("Транскрибация остановлена.")
    
    def is_recording(self):
        """Check if transcription is active."""
        return self.recording


def stream_transcribe(
    model: WhisperModel,
    device: Optional[int | str] = None,
    samplerate: int = 16_000,
    channels: int = 2,
    chunk_duration: Optional[float] = None,
    language: Optional[str] = None,
    profile: Optional[StreamProfile] = None,
    buffer_strategy: Optional[str] = None,
    buffer_pause: Optional[float] = None,
    buffer_trimming: Optional[bool] = None,
    buffer_trimming_sec: Optional[float] = None,
    callback=None,
) -> None:
    """Stream transcription in real-time with callback for results."""

    transcriber = StreamTranscriber()
    active_profile = profile or BEST_STREAM_PROFILE
    effective_chunk = chunk_duration if chunk_duration is not None else active_profile.chunk_duration

    if not transcriber.start(
        model,
        device,
        samplerate,
        channels,
        effective_chunk,
        language,
        beam_size=active_profile.beam_size,
        vad_filter=active_profile.vad_filter,
        overlap_ratio=active_profile.overlap_ratio,
        temperature=active_profile.temperature,
        use_threading=active_profile.use_threading,
        buffer_strategy=buffer_strategy or active_profile.buffer_strategy,
        buffer_pause=buffer_pause if buffer_pause is not None else active_profile.buffer_pause,
        buffer_trimming=buffer_trimming if buffer_trimming is not None else active_profile.buffer_trimming,
        buffer_trimming_sec=(
            buffer_trimming_sec if buffer_trimming_sec is not None else active_profile.buffer_trimming_sec
        ),
        callback=callback,
    ):
        return

    try:
        while transcriber.is_recording():
            time.sleep(0.1)
    except KeyboardInterrupt:
        transcriber.stop()