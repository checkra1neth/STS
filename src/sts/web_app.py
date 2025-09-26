"""Web interface for real-time speech transcription."""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
from .transcriber import load_model, StreamTranscriber

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sts-transcription-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Глобальные переменные
transcriber = StreamTranscriber()
model = None
all_transcripts = []
preloaded_models = {}  # Кэш для предзагруженных моделей


def clean_transcription_text(text: str) -> str:
    """Простая и эффективная очистка текста."""
    import re
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text.strip())
    
    if not text:
        return text
    
    # Убираем только очевидные повторы слов подряд
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Убираем только прямые повторы одного и того же слова подряд
        if (len(cleaned_words) > 0 and 
            word.lower() == cleaned_words[-1].lower() and
            len(word) > 3):  # Только для слов длиннее 3 символов
            continue
        cleaned_words.append(word)
    
    return ' '.join(cleaned_words)


def remove_overlap_with_previous(new_text: str, previous_texts: list) -> str:
    """Простое удаление очевидных пересечений."""
    if not new_text or not previous_texts:
        return new_text
    
    # Проверяем только последний сегмент на простые пересечения
    if previous_texts:
        last_text = previous_texts[-1]
        last_words = last_text.split()
        new_words = new_text.split()
        
        # Ищем простое пересечение в начале (максимум 3 слова)
        for i in range(1, min(4, len(new_words), len(last_words)) + 1):
            if (len(last_words) >= i and 
                last_words[-i:] == new_words[:i]):
                # Удаляем пересекающуюся часть
                remaining = new_words[i:]
                return ' '.join(remaining) if remaining else ""
    
    return new_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """Get available Whisper models."""
    models = [
        {'id': 'tiny', 'name': 'Tiny (только для тестов)', 'size': '~39 MB'},
        {'id': 'base', 'name': 'Base (РЕКОМЕНДУЕТСЯ)', 'size': '~74 MB'},
        {'id': 'small', 'name': 'Small (отличное качество)', 'size': '~244 MB'},
        {'id': 'medium', 'name': 'Medium (высокое качество)', 'size': '~769 MB'},
        {'id': 'large', 'name': 'Large (максимальное качество)', 'size': '~1550 MB'},
    ]
    return jsonify(models)

@app.route('/api/languages')
def get_languages():
    """Get supported languages."""
    languages = [
        {'code': 'auto', 'name': 'Автоматическое определение'},
        {'code': 'ru', 'name': 'Русский'},
        {'code': 'uk', 'name': 'Украинский'},
        {'code': 'en', 'name': 'English'},
        {'code': 'de', 'name': 'Deutsch'},
        {'code': 'fr', 'name': 'Français'},
        {'code': 'es', 'name': 'Español'},
        {'code': 'it', 'name': 'Italiano'},
        {'code': 'ja', 'name': '日本語'},
        {'code': 'zh', 'name': '中文'},
    ]
    return jsonify(languages)

@socketio.on('start_transcription')
def handle_start_transcription(data):
    global transcriber, model, all_transcripts
    
    if transcriber.is_recording():
        emit('error', {'message': 'Транскрибация уже активна'})
        return
    
    try:
        model_name = data.get('model', 'small')
        language = data.get('language', 'auto')
        speed = data.get('speed', 'fast')
        
        if language == 'auto':
            language = None
        
        # Настройки для КАЧЕСТВА + СКОРОСТИ
        speed_settings = {
            'instant': {
                'chunk_duration': 1.5,  # Увеличили для стабильности
                'cpu_threads': 4,  # Уменьшили для стабильности
                'beam_size': 5,  # Увеличили для качества
                'vad_filter': True,
                'overlap': 0.2,  # Уменьшили перекрытие
                'temperature': 0.0,
                'use_threading': True
            },
            'lightning': {
                'chunk_duration': 2.0,
                'cpu_threads': 4,
                'beam_size': 5,
                'vad_filter': True,
                'overlap': 0.15,
                'temperature': 0.0,
                'use_threading': True
            },
            'fast': {
                'chunk_duration': 2.5,
                'cpu_threads': 2,
                'beam_size': 5,
                'vad_filter': True,
                'overlap': 0.1,
                'temperature': 0.0,
                'use_threading': False
            },
            'quality': {
                'chunk_duration': 4.0,
                'cpu_threads': 2,
                'beam_size': 8,
                'vad_filter': True,
                'overlap': 0.05,
                'temperature': 0.0,
                'use_threading': False
            }
        }
        settings = speed_settings.get(speed, speed_settings['instant'])
        
        # Проверяем, есть ли модель в кэше
        model_key = f"{model_name}_{settings['cpu_threads']}"
        if model_key in preloaded_models:
            model = preloaded_models[model_key]
            emit('status', {'message': 'Модель загружена из кэша. Начинаю транскрибацию...'})
        else:
            emit('status', {'message': f'Загружаю модель {model_name}...'})
            
            # Загружаем модель с максимальной оптимизацией
            model = load_model(
                model_name,
                device='cpu',
                compute_type='int8',  # Самый быстрый тип вычислений
                cpu_threads=settings['cpu_threads']
            )
            
            # Кэшируем модель для повторного использования
            preloaded_models[model_key] = model
        
        emit('status', {'message': 'Модель загружена. Начинаю транскрибацию...'})
        
        # Очищаем предыдущие результаты
        all_transcripts = []
        
        def transcription_callback(result):
            """Callback для получения результатов транскрибации."""
            # Продвинутая постобработка
            cleaned_text = clean_transcription_text(result['text'])
            
            # Проверяем на дублирование с предыдущими сегментами
            if all_transcripts and cleaned_text:
                # Проверяем последние 2 сегмента на пересечения
                last_texts = [t['text'] for t in all_transcripts[-2:]]
                cleaned_text = remove_overlap_with_previous(cleaned_text, last_texts)
            
            if cleaned_text.strip():  # Добавляем только непустые сегменты
                transcript_data = {
                    'text': cleaned_text,
                    'language': result['language'],
                    'timestamp': result['timestamp'],
                    'duration': result['duration']
                }
                all_transcripts.append(transcript_data)
                
                # Отправляем как новый сегмент и полный текст
                full_text = ' '.join([t['text'] for t in all_transcripts])
                socketio.emit('transcription_update', {
                    'new_segment': transcript_data,
                    'full_text': full_text,
                    'total_segments': len(all_transcripts)
                })
        
        # Запускаем МГНОВЕННУЮ транскрибацию
        success = transcriber.start(
            model=model,
            device=None,  # Автоматически найдет BlackHole
            language=language,
            chunk_duration=settings['chunk_duration'],
            beam_size=settings['beam_size'],
            vad_filter=settings['vad_filter'],
            overlap_ratio=settings['overlap'],
            temperature=settings['temperature'],
            use_threading=settings['use_threading'],
            callback=transcription_callback
        )
        
        if success:
            emit('transcription_started')
        else:
            emit('error', {'message': 'Не удалось запустить транскрибацию'})
        
    except Exception as e:
        emit('error', {'message': f'Ошибка запуска: {str(e)}'})

@socketio.on('stop_transcription')
def handle_stop_transcription():
    global transcriber
    if transcriber.is_recording():
        emit('status', {'message': 'Останавливаю транскрибацию...'})
        transcriber.stop()
        emit('transcription_stopped')
    else:
        emit('error', {'message': 'Транскрибация не активна'})

@socketio.on('get_full_transcript')
def handle_get_full_transcript():
    """Отправить полный текст транскрибации."""
    full_text = ' '.join([t['text'] for t in all_transcripts])
    emit('full_transcript', {'text': full_text, 'segments': all_transcripts})

@socketio.on('clear_transcript')
def handle_clear_transcript():
    global all_transcripts
    all_transcripts = []
    emit('transcript_cleared')

def preload_models():
    """Предзагрузка моделей для МГНОВЕННОЙ работы."""
    global preloaded_models
    
    print("🚀 ПРЕДЗАГРУЖАЮ МОДЕЛИ ДЛЯ МГНОВЕННОЙ РАБОТЫ...")
    
    # ПРИОРИТЕТ: Tiny для МАКСИМАЛЬНОЙ скорости
    try:
        tiny_model = load_model('tiny', device='cpu', compute_type='int8', cpu_threads=8)
        preloaded_models['tiny_8'] = tiny_model
        print("⚡ Модель Tiny ГОТОВА К МГНОВЕННОЙ РАБОТЕ")
    except Exception as e:
        print(f"❌ Ошибка Tiny: {e}")
        
    # Base для баланса
    try:
        base_model = load_model('base', device='cpu', compute_type='int8', cpu_threads=6)
        preloaded_models['base_6'] = base_model
        print("✅ Модель Base предзагружена")
    except Exception as e:
        print(f"⚠️ Ошибка Base: {e}")

def run_web_app(host='127.0.0.1', port=8080, debug=False, preload=True):
    """Run the web application."""
    print(f"🌐 Запускаю веб-интерфейс на http://{host}:{port}")
    
    if preload:
        preload_models()
    
    print("📝 Откройте браузер и перейдите по ссылке для начала транскрибации")
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_web_app()