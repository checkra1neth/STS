# STS — легковесная транскрибация речи с Whisper

Проект демонстрирует, как запустить распознавание речи **OpenAI Whisper** на MacBook с чипом M1 Pro и 16 ГБ оперативной памяти без излишней нагрузки на систему. Мы используем библиотеку [faster-whisper](https://github.com/SYSTRAN/faster-whisper), которая содержит оптимизации для CPU и позволяет выполнять инференс в `int8`-формате.

## Установка

1. Установите зависимости Python:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Для записи аудио используется `sounddevice`, которому может потребоваться установка [PortAudio](http://www.portaudio.com/) (на macOS достаточно выполнить `brew install portaudio`).

3. Для работы с видеофайлами требуется [FFmpeg](https://ffmpeg.org/):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   ```

4. Для захвата системного аудио (стримы, браузер) на macOS установите [BlackHole](https://github.com/ExistentialAudio/BlackHole):
   ```bash
   brew install blackhole-2ch
   ```
   Затем настройте Multi-Output Device в Audio MIDI Setup для одновременного вывода на динамики и BlackHole.

5. Для получения покадровых таймингов слов используйте пакет [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) и его зависимости (включая `openai-whisper`). Установите их дополнительно:

   ```bash
   pip install whisper-timestamped openai-whisper
   ```

## Использование

Интерфейс предоставляется через модуль `sts.cli`. Его можно запускать командой:

```bash
python -m sts.cli --help
```

### Расшифровка готового файла

```bash
# Аудиофайл
python -m sts.cli --input path/to/audio.wav --output transcript.txt

# Видеофайл (аудио будет автоматически извлечено)
python -m sts.cli --input path/to/video.mp4 --language ru --output transcript.txt
```

Скрипт автоматически скачает и закеширует модель `small`, которая обеспечивает хорошее качество распознавания для десятков языков (включая русский и украинский). При необходимости можно выбрать другую модель, например `tiny` (быстрая), `base`, `medium` или `large`, указав флаг `--model`. Для английского языка доступны оптимизированные версии с суффиксом `.en` (например, `small.en`).

### Запись с микрофона в реальном времени

```bash
python -m sts.cli --record --output transcript.txt
```

### Захват системного аудио (стримы, браузер, приложения)

```bash
# Захват системного аудио до нажатия Ctrl+C
python -m sts.cli --system-audio --language ru --output transcript.txt

# Захват на определенное время (30 секунд)
python -m sts.cli --system-audio --duration 30 --language ru --output transcript.txt
```

**Как это работает:**
1. Система автоматически найдет устройство для захвата системного аудио (BlackHole на macOS)
2. Начнется запись всего аудио, воспроизводимого на компьютере
3. Нажмите `Ctrl+C` или дождитесь окончания времени для остановки
4. Результат будет транскрибирован и сохранен

### Управление аудиоустройствами

```bash
# Посмотреть все доступные устройства (включая системные)
python -m sts.cli --list-devices

# Выбрать конкретное устройство для записи
python -m sts.cli --record --input-device "MacBook Pro Microphone" --output transcript.txt

# Выбрать конкретное системное устройство
python -m sts.cli --system-audio --input-device "BlackHole 2ch" --output transcript.txt
```

### Тайминги каждого слова, субтитры и подсветка уверенности

```bash
python -m sts.cli --input meeting.wav --word-level --min-confidence 0.6 --output meeting.txt
```

Флаг `--word-level` включает использование whisper-timestamped: помимо основного текста будет сохранён JSON со списком слов, а также готовые субтитры в форматах SRT и VTT (`meeting.json`, `meeting.srt`, `meeting.vtt`).

Параметр `--min-confidence` задаёт порог подсветки слов с низкой уверенностью: в консольном выводе и субтитрах такие слова будут выделены маркерами `⟪...⟫` вместе с числовым значением. Без флага `--word-level` порог игнорируется.

### Оптимизация под Mac M1 Pro

- **CPU-инференс**: по умолчанию используется устройство `cpu`, что даёт стабильную работу без необходимости настраивать GPU.
- **Тип вычислений `int8`**: значительно снижает потребление памяти и ускоряет инференс, что особенно полезно для ноутбуков.
- **Частота дискретизации 16 кГц**: достаточна для речевых задач и помогает уменьшить размер записываемых файлов.

При необходимости можно настроить количество потоков CPU (`--cpu-threads` ограничивает количество внутренних потоков faster-whisper), размер `beam search`, температуру, язык распознавания и отключить VAD-фильтр. Если вы явно выбираете английскую сборку модели (суффикс `.en`) и указываете язык, отличный от английского, CLI автоматически переключится на многоязычный вариант.

## Структура проекта

- `src/sts/transcriber.py` — функции для записи и распознавания.
- `src/sts/cli.py` — командный интерфейс для удобного запуска.
- `requirements.txt` — список зависимостей.

## Примечание о модели

При первом запуске выбранная модель Whisper будет скачана и сохранена в кэш директории `~/.cache/faster-whisper`. Убедитесь, что у вас достаточно свободного места (для `small` требуется около 500 МБ, для `medium` — около 1.5 ГБ).
