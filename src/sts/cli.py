"""Command line interface for recording audio and transcribing it via faster-whisper."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional

import sounddevice as sd

from .transcriber import (
    capture_audio,
    load_model,
    load_word_level_model,
    transcribe_audio,
    extract_audio_from_video,
    is_video_file,
    capture_system_audio,
    find_system_audio_devices,
    stream_transcribe,
    get_best_stream_profile,
    ConfidenceMetrics,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


def _parse_input_device(device: Optional[str]) -> Optional[int | str]:
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def _should_highlight(word: TranscriptionWord, threshold: Optional[float]) -> bool:
    return (
        threshold is not None
        and word.confidence is not None
        and word.confidence < threshold
    )


def _highlight_marker(text: str) -> str:
    return f"⟪{text}⟫"


def _format_word_text(
    word: TranscriptionWord,
    threshold: Optional[float],
    marker,
) -> str:
    raw_text = word.text or ""
    if _should_highlight(word, threshold):
        if word.confidence is not None:
            return marker(f"{raw_text} ({word.confidence:.2f})")
        return marker(raw_text)
    return raw_text


def _render_segment_text(
    segment: TranscriptionSegment,
    threshold: Optional[float],
    marker=_highlight_marker,
) -> str:
    if not segment.words or threshold is None:
        return segment.text

    pieces: list[str] = []
    previous_raw = ""

    for word in segment.words:
        raw_text = word.text or ""
        if not raw_text:
            continue

        formatted = _format_word_text(word, threshold, marker)

        needs_space = (
            bool(pieces)
            and not raw_text.startswith((
                " ",
                "\n",
                "\t",
                "-",
                "–",
                "—",
                ")",
                ",",
                ".",
                "!",
                "?",
                ";",
                ":",
            ))
            and not previous_raw.endswith((
                " ",
                "\n",
                "\t",
                "-",
                "–",
                "—",
                "(",
                "«",
                "„",
            ))
        )

        if needs_space:
            pieces.append(" ")

        pieces.append(formatted)
        previous_raw = raw_text

    rendered = "".join(pieces).strip()
    return rendered if rendered else segment.text


def _result_to_json_dict(result: TranscriptionResult) -> dict:
    payload: dict[str, object] = {
        "text": result.text,
        "duration": result.duration,
        "language": result.language,
        "segments": [],
    }

    segments: list[dict[str, object]] = []
    for index, segment in enumerate(result.segments, start=1):
        entry: dict[str, object] = {
            "index": index,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        if segment.confidence is not None:
            entry["confidence"] = segment.confidence
        if segment.avg_logprob is not None:
            entry["avg_logprob"] = segment.avg_logprob
        if segment.compression_ratio is not None:
            entry["compression_ratio"] = segment.compression_ratio
        if segment.no_speech_prob is not None:
            entry["no_speech_prob"] = segment.no_speech_prob
        if segment.words:
            entry["words"] = [
                {
                    "start": word.start,
                    "end": word.end,
                    "text": word.text,
                    **(
                        {"confidence": word.confidence}
                        if word.confidence is not None
                        else {}
                    ),
                }
                for word in segment.words
            ]
        segments.append(entry)

    payload["segments"] = segments

    if result.words:
        payload["words"] = [
            {
                "start": word.start,
                "end": word.end,
                "text": word.text,
                **(
                    {"confidence": word.confidence}
                    if word.confidence is not None
                    else {}
                ),
            }
            for word in result.words
        ]

    if isinstance(result.word_confidence, ConfidenceMetrics):
        payload["word_confidence"] = {
            "average": result.word_confidence.average,
            "minimum": result.word_confidence.minimum,
            "maximum": result.word_confidence.maximum,
            "count": result.word_confidence.count,
        }

    return payload


def _format_timestamp(value: float, *, separator: str) -> str:
    total_milliseconds = max(0, int(round(value * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"


def _build_srt(result: TranscriptionResult, threshold: Optional[float]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        lines.append(str(index))
        start = _format_timestamp(segment.start, separator=",")
        end = _format_timestamp(segment.end, separator=",")
        lines.append(f"{start} --> {end}")
        lines.append(_render_segment_text(segment, threshold))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_vtt(result: TranscriptionResult, threshold: Optional[float]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for segment in result.segments:
        start = _format_timestamp(segment.start, separator=".")
        end = _format_timestamp(segment.end, separator=".")
        lines.append(f"{start} --> {end}")
        lines.append(_render_segment_text(segment, threshold))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _compose_highlighted_transcript(
    result: TranscriptionResult,
    threshold: Optional[float],
) -> str:
    if threshold is None:
        return result.text

    parts = [
        _render_segment_text(segment, threshold)
        for segment in result.segments
    ]
    return "\n".join(filter(None, parts))


def _print_confidence_summary(summary: Optional[ConfidenceMetrics]) -> None:
    if not summary:
        return

    print(
        "Средняя уверенность слов: "
        f"{summary.average:.2f} (мин: {summary.minimum:.2f}, макс: {summary.maximum:.2f}, всего: {summary.count})"
    )


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
        help=(
            "Файл для сохранения результата распознавания. В word-level режиме дополнительно "
            "создаются JSON/SRT/VTT с таймкодами слов."
        ),
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
        "--word-level",
        action="store_true",
        help=(
            "Использовать whisper-timestamped для покадровой разбивки и таймингов слов."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "Минимальная уверенность слова для подсветки в выводе и субтитрах "
            "(работает с --word-level)."
        ),
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

    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence ожидает значение в диапазоне от 0 до 1.")

    if args.min_confidence is not None and not args.word_level:
        print(
            "⚠️  --min-confidence применяется только вместе с --word-level, параметр будет проигнорирован.",
            file=sys.stderr,
        )
        args.min_confidence = None

    # Проверяем, что выбран один из источников
    if not any([args.input, args.record, args.system_audio, args.web, args.stream]):
        parser.error("Необходимо выбрать один из: --input, --record, --system-audio, --web, или --stream")

    if args.word_level and (args.web or args.stream):
        parser.error("Режим --word-level доступен только для офлайн транскрибации файлов или записей.")
    
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

        profile = get_best_stream_profile()
        cpu_threads = args.cpu_threads if args.cpu_threads is not None else profile.cpu_threads

        model = load_model(
            model_name,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=cpu_threads,
        )

        def print_result(result):
            print(f"[{result['language']}] {result['text']}")
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

        timestamped_model = None
        if args.word_level:
            try:
                timestamped_model = load_word_level_model(
                    model_name,
                    device=args.device,
                )
            except ImportError as exc:
                print(f"Не удалось загрузить whisper-timestamped: {exc}", file=sys.stderr)
                raise SystemExit(1)
            except Exception as exc:
                print(f"Ошибка инициализации whisper-timestamped: {exc}", file=sys.stderr)
                raise SystemExit(1)

        result = transcribe_audio(
            model,
            audio_path,
            beam_size=args.beam_size,
            language=requested_language,
            temperature=args.temperature,
            vad_filter=not args.no_vad,
            word_level=args.word_level,
            timestamped_model=timestamped_model,
            timestamped_model_name=model_name,
            timestamped_device=args.device,
        )

        print("\n=== Результат распознавания ===")
        if result.language:
            print(f"Определённый язык: {result.language}")
        print(f"Длительность: {result.duration:.2f} с")
        highlight_threshold = args.min_confidence if args.word_level else None
        if result.word_confidence:
            _print_confidence_summary(result.word_confidence)
        if args.word_level and highlight_threshold is not None:
            print(f"Порог подсветки: {highlight_threshold:.2f}")
        print()

        for idx, segment in enumerate(result.segments, start=1):
            segment_text = _render_segment_text(segment, highlight_threshold)
            print(f"[{idx:03d}] {segment.start:7.2f} — {segment.end:7.2f}: {segment_text}")

        print("\nИтоговый текст:\n")
        final_text = (
            _compose_highlighted_transcript(result, highlight_threshold)
            if args.word_level
            else result.text
        )
        print(final_text)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            text_payload = final_text if args.word_level else result.text
            args.output.write_text(text_payload, encoding="utf-8")
            print(f"\nТранскрипция сохранена в {args.output}")

            if args.word_level:
                json_path = args.output.with_suffix(".json")
                srt_path = args.output.with_suffix(".srt")
                vtt_path = args.output.with_suffix(".vtt")

                json_payload = _result_to_json_dict(result)
                json_path.write_text(
                    json.dumps(json_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                srt_path.write_text(_build_srt(result, highlight_threshold), encoding="utf-8")
                vtt_path.write_text(_build_vtt(result, highlight_threshold), encoding="utf-8")

                print(
                    "Дополнительно сохранены файлы: "
                    f"{json_path.name}, {srt_path.name}, {vtt_path.name}"
                )
    finally:
        if is_temp_file and audio_path and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
