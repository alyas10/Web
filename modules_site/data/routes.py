# modules_site/data/routes.py
from flask import render_template, request, jsonify, redirect, url_for, session
from flask import current_app  # <-- Импортируем current_app для доступа к объектам из app
from . import bp  # Импортируем свой Blueprint
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from collections import Counter
import uuid
import os

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

# SAMPLE_DATA для предпросмотра (можно вынести в отдельный файл, если нужно)
SAMPLE_DATA = [
        {'id': 1, 'timestamp': '2026-02-07 10:15:32', 'src_ip': '192.168.1.105', 'dst_ip': '8.8.8.8', 'protocol': 'TCP',
         'port': 443, 'bytes': 1452, 'label': 'Benign'},
        {'id': 2, 'timestamp': '2026-02-07 10:15:33', 'src_ip': '192.168.1.105', 'dst_ip': '203.0.113.45',
         'protocol': 'UDP', 'port': 53, 'bytes': 128, 'label': 'Benign'},
        {'id': 3, 'timestamp': '2026-02-07 10:15:34', 'src_ip': '10.0.0.23', 'dst_ip': '192.168.1.105',
         'protocol': 'TCP', 'port': 22, 'bytes': 2048, 'label': 'Intrusion'},
        {'id': 4, 'timestamp': '2026-02-07 10:15:35', 'src_ip': '172.16.0.88', 'dst_ip': '192.168.1.105',
         'protocol': 'ICMP', 'port': 0, 'bytes': 512, 'label': 'DoS'},
        {'id': 5, 'timestamp': '2026-02-07 10:15:36', 'src_ip': '192.168.1.105', 'dst_ip': '1.1.1.1', 'protocol': 'TCP',
         'port': 80, 'bytes': 896, 'label': 'Benign'},
        {'id': 6, 'timestamp': '2026-02-07 10:15:37', 'src_ip': '10.0.0.45', 'dst_ip': '192.168.1.105',
         'protocol': 'TCP', 'port': 3389, 'bytes': 4096, 'label': 'Anomaly'},
    ]

@bp.route('/')
def data_upload():
    """Роут для страницы /data"""
    return render_template('data_upload.html', sample_data=SAMPLE_DATA, uploaded_file=None)

@bp.route('/upload', methods=['POST'])
def upload_file():
    """Роут для загрузки файла POST /data/upload"""
    # Получаем объекты из current_app
    model_manager = current_app.model_manager
    data_adapter = current_app.data_adapter
    visualizer = current_app.visualizer
    feature_info = current_app.feature_info

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    original_filename = file.filename
    filename = secure_filename(original_filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['last_uploaded_filename'] = filename

    try:
        from data_loader.csv_loader import CSVDataLoader
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.csv':
            loader = CSVDataLoader()
        else:
            return jsonify({'error': f'Формат {ext} не поддерживается'}), 400

        raw_df = loader.load(filepath)
        visualization_cards_html = visualizer.generate_overview_plots(raw_df)

        processed_df = data_adapter.prepare(raw_df)
        sample_data = processed_df.head(6).to_dict(orient='records')
        columns = processed_df.columns.tolist()

        session_id = str(uuid.uuid4())
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{session_id}_processed.pkl")
        raw_df.to_pickle(temp_path)

        uploaded_file_info = {
            'name': filename,
            'size': os.path.getsize(filepath),
            'session_id': session_id
        }

        return render_template('data_upload.html',
                               sample_data=sample_data,
                               columns=columns,
                               uploaded_file=uploaded_file_info,
                               visualization_cards=visualization_cards_html)

    except Exception as e:
        return jsonify({'error': f"Ошибка обработки данных: {str(e)}"}), 500

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

        return redirect(url_for('dashboard.dashboard'))  # ← Обратите внимание на имя: 'dashboard.dashboard'

    except Exception as e:
        return jsonify({"error": f"Ошибка анализа: {str(e)}"}), 500