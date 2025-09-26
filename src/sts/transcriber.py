"""Utility helpers for capturing audio and transcribing it with faster-whisper."""

from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    model_size: str = "small",
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
        # The faster-whisper bindings expose thread configuration via num_workers
        # parameters. Limiting the thread count allows the caller to avoid
        # saturating the entire CPU on smaller machines.
        kwargs["num_workers"] = cpu_threads

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
        
    def start(self, model: WhisperModel, device: Optional[int | str] = None,
              samplerate: int = 16_000, channels: int = 2, chunk_duration: float = 5.0,
              language: Optional[str] = None, beam_size: int = 5, vad_filter: bool = True,
              overlap_ratio: float = 0.25, temperature: float = 0.0, 
              use_threading: bool = True, callback=None):
        """Start streaming transcription."""
        
        if self.recording:
            return False
            
        import tempfile
        import time
        import threading
        from pathlib import Path
        
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
        
        # Создаем пул потоков для обработки
        import concurrent.futures
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads)
        
        def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
            if status:
                print(status, file=sys.stderr)
            if self.recording:
                self.audio_queue.put(indata.copy())
        
        def process_audio_chunks():
            """Process audio chunks in separate thread."""
            chunk_data = []
            chunk_start_time = time.time()
            
            while self.recording or not self.audio_queue.empty():
                try:
                    # Получаем аудиоданные с таймаутом
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    chunk_data.append(audio_chunk)
                    
                    # Проверяем, прошло ли достаточно времени для обработки чанка
                    current_time = time.time()
                    if current_time - chunk_start_time >= chunk_duration:
                        if chunk_data and self.recording:
                            # Создаем копию данных для обработки в отдельном потоке
                            chunk_to_process = chunk_data.copy()
                            
                            # МГНОВЕННАЯ обработка через пул потоков
                            def process_chunk_instant():
                                try:
                                    combined_chunk = np.concatenate(chunk_to_process, axis=0)
                                    
                                    # Прямая обработка в памяти БЕЗ файлов
                                    import io
                                    buffer = io.BytesIO()
                                    with sf.SoundFile(buffer, mode='w', samplerate=samplerate, 
                                                    channels=channels, format='WAV') as file:
                                        file.write(combined_chunk)
                                    
                                    buffer.seek(0)
                                    
                                    # СУПЕР-БЫСТРАЯ транскрибация
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                                        tmp_file.write(buffer.getvalue())
                                        tmp_file.flush()
                                        tmp_path = Path(tmp_file.name)
                                        
                                        # МИНИМАЛЬНЫЕ настройки для МАКСИМАЛЬНОЙ скорости
                                        result = transcribe_audio(
                                            model,
                                            tmp_path,
                                            language=language,
                                            beam_size=beam_size,
                                            temperature=temperature,
                                            vad_filter=vad_filter,
                                        )
                                        
                                        # МГНОВЕННЫЙ callback
                                        if callback and result.text.strip() and self.recording:
                                            callback({
                                                'text': result.text.strip(),
                                                'language': result.language,
                                                'timestamp': current_time,
                                                'duration': result.duration
                                            })
                                
                                except Exception as e:
                                    if self.recording:
                                        print(f"Ошибка: {e}")
                            
                            # Отправляем в пул потоков для МГНОВЕННОЙ обработки
                            if use_threading:
                                self.thread_pool.submit(process_chunk_instant)
                            else:
                                process_chunk_instant()
                        
                        # Умное перекрытие для предотвращения дублей
                        if overlap_ratio > 0 and len(chunk_data) > 10:  # Минимум данных для перекрытия
                            overlap_size = int(len(chunk_data) * overlap_ratio)
                            # Ограничиваем размер перекрытия
                            overlap_size = min(overlap_size, len(chunk_data) // 2)
                            chunk_data = chunk_data[-overlap_size:] if overlap_size > 0 else []
                        else:
                            chunk_data = []  # Полная очистка если мало данных
                        chunk_start_time = current_time
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    if self.recording:
                        print(f"Ошибка обработки аудио: {e}")
                    break
        
        print("Начинаю потоковую транскрибацию.")
        
        # Запускаем обработку аудио в отдельном потоке
        self.processing_thread = threading.Thread(target=process_audio_chunks)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
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
        
        print("Транскрибация остановлена.")
    
    def is_recording(self):
        """Check if transcription is active."""
        return self.recording


def stream_transcribe(
    model: WhisperModel,
    device: Optional[int | str] = None,
    samplerate: int = 16_000,
    channels: int = 2,
    chunk_duration: float = 5.0,
    language: Optional[str] = None,
    callback=None,
) -> None:
    """Stream transcription in real-time with callback for results."""
    
    transcriber = StreamTranscriber()
    
    if not transcriber.start(model, device, samplerate, channels, chunk_duration, language, callback):
        return
    
    try:
        while transcriber.is_recording():
            time.sleep(0.1)
    except KeyboardInterrupt:
        transcriber.stop()