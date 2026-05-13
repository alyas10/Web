# modules_site/data/routes.py
from flask import render_template, request, jsonify, redirect, url_for, session
from flask import current_app
from . import bp
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from collections import Counter
import uuid
import os
import threading

# Глобальный словарь для хранения прогресса (ключ: session_id, значение: %)
processing_progress = {}
processing_results = {}

# Функция для определения цвета метки (копия из app.py)
def _label_color(label: str) -> str:
    l = (label or "").strip().lower()
    if l == "benign":
        return "green"
    if "dos" in l or "ddos" in l:
        return "red"
    if "intrusion" in l:
        return "orange"
    if "anomaly" in l:
        return "yellow"
    return "gray"


# SAMPLE_DATA для предпросмотра
SAMPLE_DATA = [
    {'id': 1, 'timestamp': '2026-02-07 10:15:32', 'src_ip': '192.168.1.105', 'dst_ip': '8.8.8.8', 'protocol': 'TCP',
     'port': 443, 'bytes': 1452, 'label': 'Benign'},
    {'id': 2, 'timestamp': '2026-02-07 10:15:33', 'src_ip': '192.168.1.105', 'dst_ip': '203.0.113.45',
     'protocol': 'UDP', 'port': 53, 'bytes': 128, 'label': 'Benign'},
    {'id': 3, 'timestamp': '2026-02-07 10:15:34', 'src_ip': '10.0.0.23', 'dst_ip': '192.168.1.105', 'protocol': 'TCP',
     'port': 22, 'bytes': 2048, 'label': 'Intrusion'},
    {'id': 4, 'timestamp': '2026-02-07 10:15:35', 'src_ip': '172.16.0.88', 'dst_ip': '192.168.1.105',
     'protocol': 'ICMP', 'port': 0, 'bytes': 512, 'label': 'DoS'},
    {'id': 5, 'timestamp': '2026-02-07 10:15:36', 'src_ip': '192.168.1.105', 'dst_ip': '1.1.1.1', 'protocol': 'TCP',
     'port': 80, 'bytes': 896, 'label': 'Benign'},
    {'id': 6, 'timestamp': '2026-02-07 10:15:37', 'src_ip': '10.0.0.45', 'dst_ip': '192.168.1.105', 'protocol': 'TCP',
     'port': 3389, 'bytes': 4096, 'label': 'Anomaly'},
]


@bp.route('/')
def data_upload():
    """Роут для страницы /data"""
    # Проверяем, есть ли session_id в запросе (после завершения обработки)
    session_id = request.args.get('session_id')

    if session_id and session_id in processing_results:
        # Берем результаты из кэша
        results = processing_results[session_id]
        # Очищаем кэш
        processing_results.pop(session_id, None)
        processing_progress.pop(session_id, None)

        return render_template(
            'data_upload.html',
            sample_data=results['sample_data'],
            columns=results['columns'],
            uploaded_file=results['uploaded_file_info'],
            visualization_cards=results['visualization_cards'],
            data_summary=results['data_summary']
        )

    # Если session_id нет или результаты не найдены — показываем заглушки
    return render_template('data_upload.html', sample_data=SAMPLE_DATA, uploaded_file=None)


@bp.route('/upload', methods=['POST'])
def upload_file():
    """Роут для загрузки файла POST /data/upload"""
    model_manager = current_app.model_manager
    data_adapter = current_app.data_adapter
    visualizer = current_app.visualizer
    feature_info = current_app.feature_info

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        original_filename = file.filename
        filename = secure_filename(original_filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['last_uploaded_filename'] = filename

        # Генерируем уникальный ID для этой сессии обработки
        session_id = str(uuid.uuid4())
        processing_progress[session_id] = 0
        # Получаем реальный объект app для использования в фоновом потоке
        app = current_app._get_current_object()
        # Функция обратного вызова для обновления прогресса
        def update_progress(rows_processed):
            try:
                file_size = os.path.getsize(filepath)
                estimated_total_rows = max(1, file_size // 100)
                percent = min(100, int((rows_processed / estimated_total_rows) * 100))
                processing_progress[session_id] = percent
            except:
                pass

        # Функция для фоновой обработки
        def process_file_in_background():
         with app.app_context():
            try:
                from data_loader.csv_loader import CSVDataLoader
                from data_loader.pcap_loader import PcapScapyDataLoader

                ext = os.path.splitext(filename)[1].lower()
                if ext == '.csv':
                    loader = CSVDataLoader()
                    chunksize = 100000
                elif ext in PcapScapyDataLoader().supported_extensions:
                    loader = PcapScapyDataLoader()
                    chunksize = 10000
                else:
                    processing_progress[session_id] = -1
                    return

                # Загружаем с чанками и колбэком прогресса
                raw_df = loader.load(filepath, chunksize=chunksize, progress_callback=update_progress)

                # После загрузки - визуализация и подготовка
                visualization_cards_html = visualizer.generate_overview_plots(raw_df)
                data_summary = visualizer.generate_data_summary(raw_df)

                processed_df = data_adapter.prepare(raw_df)
                sample_data = processed_df.head(6).to_dict(orient='records')
                columns = processed_df.columns.tolist()

                # Сохраняем результаты в сессию
                temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{session_id}_processed.pkl")
                raw_df.to_pickle(temp_path)

                session_data = {
                    'sample_data': sample_data,
                    'columns': columns,
                    'visualization_cards': visualization_cards_html,
                    'data_summary': data_summary,
                    'uploaded_file_info': {
                        'name': filename,
                        'size': os.path.getsize(filepath),
                        'session_id': session_id
                    }
                }
                processing_results[session_id] = session_data
                processing_progress[session_id] = 100

            except Exception as e:
                print(f"Ошибка в фоновой обработке: {e}")
                processing_progress[session_id] = -1

        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(target=process_file_in_background)
        thread.daemon = True
        thread.start()

        # Сразу возвращаем session_id, чтобы фронтенд мог начать поллинг
        return jsonify({
            'status': 'started',
            'session_id': session_id,
            'message': 'Файл загружен. Начинается обработка...'
        })

    except Exception as e:
        return jsonify({'error': f"Ошибка обработки данных: {str(e)}"}), 500


@bp.route('/progress/<session_id>')
def get_progress(session_id):
    """Роут для получения статуса обработки"""
    percent = processing_progress.get(session_id, 0)

    if percent == 100:
        # При первом запросе "completed" — очищаем прогресс, но оставляем результаты
        processing_progress.pop(session_id, None)
        return jsonify({'status': 'completed', 'percent': 100})
    elif percent == -1:
        processing_progress.pop(session_id, None)
        processing_results.pop(session_id, None)
        return jsonify({'status': 'error', 'percent': 0, 'message': 'Ошибка обработки'}), 500
    else:
        return jsonify({'status': 'processing', 'percent': percent})


@bp.route('/results/<session_id>')
def get_results(session_id):
    """Роут для получения статуса (не удаляет результаты)"""
    results = processing_results.get(session_id)

    if results is None:
        return jsonify({'error': 'Результаты не найдены или уже удалены'}), 404

    # Возвращаем результаты, но НЕ удаляем их — они понадобятся при редиректе
    return jsonify({
        'status': 'success',
        'data': results
    })


# ... после функции get_results() ...

@bp.route('/files')
def list_uploaded_files():
    """Роут для получения списка загруженных файлов"""
    upload_folder = current_app.config['UPLOAD_FOLDER']

    if not os.path.exists(upload_folder):
        return jsonify({'files': []})

    files = []
    for filename in os.listdir(upload_folder):
        filepath = os.path.join(upload_folder, filename)
        # Пропускаем временные файлы и обработанные pickle-файлы
        if (filename.endswith('.pkl') or
                filename.startswith('.') or
                filename.endswith('_processed.pkl')):
            continue

        try:
            file_stat = os.stat(filepath)
            files.append({
                'name': filename,
                'size': file_stat.st_size,
                'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'ext': os.path.splitext(filename)[1].lower()
            })
        except Exception:
            continue

    # Сортируем по дате изменения (новые сверху)
    files.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({'files': files})

@bp.route('/delete_file/<filename>', methods=['POST'])
def delete_file(filename):
    """Роут для удаления файла из списка загруженных"""
    filename = secure_filename(filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 404

    try:
        os.remove(filepath)
        # Также удаляем связанный processed файл если есть
        session_id = filename.replace('.csv', '').replace('.pcap', '').replace('.pcapng', '')
        processed_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{session_id}_processed.pkl")
        if os.path.exists(processed_path):
            os.remove(processed_path)

        return jsonify({'status': 'success', 'message': f'Файл {filename} удален'})
    except Exception as e:
        return jsonify({'error': f'Ошибка удаления файла: {str(e)}'}), 500

@bp.route('/select_file/<filename>', methods=['POST'])
def select_file(filename):
    """Роут для выбора файла для анализа"""
    filename = secure_filename(filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Файл не найден'}), 404

    # Сохраняем выбранный файл в сессии
    session['last_uploaded_filename'] = filename

    # Генерируем preview для отображения (опционально)
    try:
        from data_loader.csv_loader import CSVDataLoader
        from data_loader.pcap_loader import PcapScapyDataLoader

        ext = os.path.splitext(filename)[1].lower()
        if ext == '.csv':
            loader = CSVDataLoader()
        elif ext in PcapScapyDataLoader().supported_extensions:
            loader = PcapScapyDataLoader()
        else:
            return jsonify({'error': f'Формат {ext} не поддерживается'}), 400

        # Загружаем только первые 6 строк для превью
        raw_df = loader.load(filepath)
        processed_df = current_app.data_adapter.prepare(raw_df)
        sample_data = processed_df.head(6).to_dict(orient='records')
        columns = processed_df.columns.tolist()

        return jsonify({
            'status': 'success',
            'filename': filename,
            'sample_data': sample_data,
            'columns': columns,
            'file_info': {
                'name': filename,
                'size': os.path.getsize(filepath),
                'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
            }
        })

    except Exception as e:
        return jsonify({'error': f'Ошибка загрузки файла: {str(e)}'}), 500

@bp.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Роут для запуска анализа POST /data/start_analysis"""
    model_manager = current_app.model_manager
    data_adapter = current_app.data_adapter
    visualizer = current_app.visualizer

    filename = request.form.get('filename') or session.get('last_uploaded_filename')
    if not filename:
        return jsonify({"error": "Не передано имя файла для анализа."}), 400

    filename = secure_filename(filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({"error": f"Файл не найден в uploads: {filename}"}), 404

    ext = os.path.splitext(filename)[1].lower()
    if ext != ".csv":
        return jsonify({"error": f"Формат {ext} не поддерживается для анализа (ожидается .csv)."}), 400

    try:
        from data_loader.csv_loader import CSVDataLoader
        loader = CSVDataLoader()
        raw_df = loader.load(filepath)
        processed_df = data_adapter.prepare(raw_df)

        algo = request.form.get('algo', 'lightgbm')
        env = request.form.get('env', 'test')

        if algo not in model_manager.file_map:
            return jsonify({"error": f"Модель '{algo}' не поддерживается."}), 400

        predictions = model_manager.predict(algo=algo, data=processed_df, env=env)

        total = len(predictions)
        counts = Counter(predictions)
        benign = counts.get("Benign", counts.get("benign", 0))
        threats = total - benign

        distribution = []
        for label, cnt in counts.most_common():
            pct = round((cnt / total) * 100, 2) if total else 0.0
            distribution.append({
                "type": label,
                "count": cnt,
                "percentage": pct,
                "color": _label_color(label),
            })

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        analysis_results = {
            "filename": filename,
            "timestamp": ts,
            "model_used": f"{algo}/{env}",
            "rows": total,
            "threats": threats,
            "class_counts": dict(counts),
            "threat_distribution": distribution,
            "predictions_sample": predictions[:200],
        }
        session['analysis_results'] = analysis_results

        history = session.get("recent_analyses", [])
        history.insert(0, {
            "model": algo,
            "dataset": filename,
            "accuracy": "-",
            "threats": threats,
            "timestamp": ts.replace("T", " "),
        })
        session["recent_analyses"] = history[:10]

        return redirect(url_for('dashboard.dashboard'))

    except Exception as e:
        return jsonify({"error": f"Ошибка анализа: {str(e)}"}), 500
