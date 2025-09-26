"""Command line interface for recording audio and transcribing it via faster-whisper."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Optional

import sounddevice as sd

from .transcriber import (
    capture_audio,
    load_model,
    transcribe_audio,
    extract_audio_from_video,
    is_video_file,
    capture_system_audio,
    find_system_audio_devices,
    stream_transcribe,
    get_best_stream_profile,
)


def _parse_input_device(device: Optional[str]) -> Optional[int | str]:
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Запись аудио с микрофона или работа с готовым файлом с последующей транскрибацией "
            "через оптимизированную сборку Whisper."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--input",
        type=Path,
        help="Путь к аудио- или видеофайлу (поддерживаются форматы ffmpeg).",
    )
    source_group.add_argument(
        "--record",
        action="store_true",
        help="Записать аудио с микрофона и использовать его для распознавания.",
    )
    source_group.add_argument(
        "--system-audio",
        action="store_true",
        help="Записать системное аудио (стримы, браузер, приложения) для распознавания.",
    )
    source_group.add_argument(
        "--web",
        action="store_true",
        help="Запустить веб-интерфейс для транскрибации в реальном времени.",
    )
    source_group.add_argument(
        "--stream",
        action="store_true",
        help="Потоковая транскрибация в реальном времени (консольный режим).",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Размер модели Whisper. small — оптимальный баланс качества и скорости для CPU и поддерживает многие языки.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство для вычислений (cpu, cuda, metal). Для Mac M1 лучше оставить cpu.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "Тип вычислений faster-whisper (int8, int8_float16, float16, float32, auto). "
            "Значение auto выбирает оптимальный баланс скорости и качества."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Количество потоков CPU. По умолчанию faster-whisper выберет автоматически.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16_000,
        help="Частота дискретизации при записи (по умолчанию 16 кГц).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Количество каналов при записи. Whisper ожидает моно, поэтому оставьте 1.",
    )
    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="Идентификатор устройства записи sounddevice (номер или название).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Вывести список доступных аудиоустройств и завершить работу.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Длительность записи в секундах (только для --system-audio). Если не указано, запись до Ctrl+C.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Файл для сохранения результата распознавания (текстовый формат).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Размер beam search. Значения 1-5 ускоряют инференс ценой качества.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Язык речи (например, 'ru' или 'en'). Если не задан, модель попытается определить его автоматически.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Температура выборки. Оставьте 0.0 для детерминированного результата.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Отключить VAD-фильтр быстрее-шёпота (по умолчанию включён).",
    )
    parser.add_argument(
        "--stabilize-stream",
        action="store_true",
        help="Включить стабилизацию слов и таймкодов в потоковом режиме.",
    )
    parser.add_argument(
        "--stabilize-confirmation-window",
        type=int,
        default=None,
        help="Количество гипотез, которые должны совпасть, прежде чем слово будет подтверждено.",
    )
    return parser


def _maybe_list_devices(list_requested: bool) -> None:
    if not list_requested:
        return
    
    print("=== Все аудиоустройства ===")
    print(sd.query_devices())
    
    print("\n=== Системные аудиоустройства (для захвата стримов) ===")
    system_devices = find_system_audio_devices()
    if system_devices:
        for device in system_devices:
            print(f"[{device['index']}] {device['name']} ({device['channels']} каналов)")
    else:
        print("Системные аудиоустройства не найдены.")
        print("Для macOS установите BlackHole: brew install blackhole-2ch")
    
    raise SystemExit(0)


def _resolve_input_file(args: argparse.Namespace) -> tuple[Path, bool]:
    """Resolve input file and return (path, is_temp_file)."""
    if args.input:
        path = args.input.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        
        # Если это видеофайл, извлекаем аудио
        if is_video_file(path):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            extract_audio_from_video(path, tmp_path, args.samplerate)
            return tmp_path, True
        
        return path, False

    # Запись с микрофона или системного аудио
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    
    if args.system_audio:
        # Системное аудио обычно стерео
        channels = 2 if args.channels == 1 else args.channels
        capture_system_audio(
            tmp_path,
            samplerate=args.samplerate,
            channels=channels,
            device=_parse_input_device(args.input_device),
            duration=args.duration,
        )
    else:
        # Обычная запись с микрофона
        assert args.record
        capture_audio(
            tmp_path,
            samplerate=args.samplerate,
            channels=args.channels,
            device=_parse_input_device(args.input_device),
        )
    
    return tmp_path, True


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _maybe_list_devices(args.list_devices)
    
    # Проверяем, что выбран один из источников
    if not any([args.input, args.record, args.system_audio, args.web, args.stream]):
        parser.error("Необходимо выбрать один из: --input, --record, --system-audio, --web, или --stream")
    
    # Запуск веб-интерфейса
    if args.web:
        from .web_app import run_web_app
        run_web_app(host='127.0.0.1', port=8080, debug=False)
        return
    
    # Потоковая транскрибация
    if args.stream:
        requested_language = args.language.lower() if args.language else None
        model_name = args.model

        if requested_language and requested_language != "en" and model_name.endswith(".en"):
            multilingual_candidate = model_name[: -len(".en")]
            if multilingual_candidate:
                print(
                    "Выбранная модель поддерживает только английский. Для распознавания языка "
                    f"'{requested_language}' автоматически использую '{multilingual_candidate}'.",
                    file=sys.stderr,
                )
                model_name = multilingual_candidate

        if args.stabilize_confirmation_window is not None and args.stabilize_confirmation_window < 1:
            print(
                "Значение --stabilize-confirmation-window должно быть не меньше 1.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        profile = get_best_stream_profile()
        if args.stabilize_stream or args.stabilize_confirmation_window is not None:
            profile = replace(
                profile,
                stabilize_stream=args.stabilize_stream or profile.stabilize_stream,
                stabilize_confirmation_window=(
                    args.stabilize_confirmation_window
                    if args.stabilize_confirmation_window is not None
                    else profile.stabilize_confirmation_window
                ),
            )

        cpu_threads = args.cpu_threads if args.cpu_threads is not None else profile.cpu_threads

        model = load_model(
            model_name,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=cpu_threads,
        )

        def print_result(result):
            language = result.get('language') or '--'
            print(f"[{language}] {result['text']}")

            confirmed_words = result.get('words') or []
            if confirmed_words:
                timing_parts = []
                for word in confirmed_words:
                    start = word.get('start')
                    end = word.get('end')
                    text = word.get('text', '').strip()
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and text:
                        timing_parts.append(f"{text} ({start:.2f}-{end:.2f})")
                if timing_parts:
                    print("   ↳ " + ", ".join(timing_parts))

            if args.output:
                with open(args.output, 'a', encoding='utf-8') as f:
                    f.write(f"{result['text']} ")

        stream_transcribe(
            model=model,
            device=_parse_input_device(args.input_device),
            language=requested_language,
            profile=profile,
            callback=print_result
        )
        return

    audio_path: Optional[Path] = None
    is_temp_file = False
    requested_language = args.language.lower() if args.language else None
    model_name = args.model

    if requested_language and requested_language != "en" and model_name.endswith(".en"):
        multilingual_candidate = model_name[: -len(".en")]
        if multilingual_candidate:
            print(
                "Выбранная модель поддерживает только английский. Для распознавания языка "
                f"'{requested_language}' автоматически использую '{multilingual_candidate}'.",
                file=sys.stderr,
            )
            model_name = multilingual_candidate
        else:
            print(
                "Выбранная модель поддерживает только английский. Укажите многоязычную модель (например, small).",
                file=sys.stderr,
            )
            raise SystemExit(1)

    try:
        audio_path, is_temp_file = _resolve_input_file(args)

        model = load_model(
            model_name,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )

        if requested_language and requested_language != "en" and not getattr(model, "is_multilingual", True):
            print(
                "Загруженная модель поддерживает только английский язык. Выберите многоязычную модель (например, small).",
                file=sys.stderr,
            )
            raise SystemExit(1)

        result = transcribe_audio(
            model,
            audio_path,
            beam_size=args.beam_size,
            language=requested_language,
            temperature=args.temperature,
            vad_filter=not args.no_vad,
        )

        print("\n=== Результат распознавания ===")
        if result.language:
            print(f"Определённый язык: {result.language}")
        print(f"Длительность: {result.duration:.2f} с")
        print()

        for idx, segment in enumerate(result.segments, start=1):
            print(f"[{idx:03d}] {segment.start:7.2f} — {segment.end:7.2f}: {segment.text}")

        print("\nИтоговый текст:\n")
        print(result.text)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.text, encoding="utf-8")
            print(f"\nТранскрипция сохранена в {args.output}")
    finally:
        if is_temp_file and audio_path and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
