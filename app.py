from flask import (Flask, render_template, request,
                   jsonify, redirect, url_for,session)

from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from collections import Counter
import joblib
import pandas as pd ,numpy as np
import os,shutil
from model_manager.model_manager import ModelManager
from model_manager.model_utils  import NumericFeatureSelector
from data_loader.base import DataPipelineAdapter
from data_loader.csv_loader import CSVDataLoader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

def clean_upload_folder():
    """Очищает папку загрузок, удаляя и пересоздавая её."""
    folder = app.config['UPLOAD_FOLDER']
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print("Папка uploads очищена.")
    except Exception as e:
        print(f" Ошибка очистки папки: {e}")

clean_upload_folder()

# Тест загрузки пайплайна
try:
    pipe = joblib.load("models/lightgbm/test/full_pipeline.pkl")
    feature_info = joblib.load('models/lightgbm/test/feature_info.pkl')
    REQUIRED_FEATURES = feature_info['numeric_features'] + feature_info['categorical_features']
    print(feature_info['numeric_features':20])
    print(feature_info['categorical_features':20])
except Exception as e:
    pipe, REQUIRED_FEATURES = None, None


# Инициализируем менеджер моделей
model_manager = ModelManager(models_root="models")
data_adapter = DataPipelineAdapter(expected_features=REQUIRED_FEATURES)
#model_manager.clear_cache()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Sample data for demonstration
SAMPLE_DATA = [
    {'id': 1, 'timestamp': '2026-02-07 10:15:32', 'src_ip': '192.168.1.105', 'dst_ip': '8.8.8.8', 'protocol': 'TCP', 'port': 443, 'bytes': 1452, 'label': 'Benign'},
    {'id': 2, 'timestamp': '2026-02-07 10:15:33', 'src_ip': '192.168.1.105', 'dst_ip': '203.0.113.45', 'protocol': 'UDP', 'port': 53, 'bytes': 128, 'label': 'Benign'},
    {'id': 3, 'timestamp': '2026-02-07 10:15:34', 'src_ip': '10.0.0.23', 'dst_ip': '192.168.1.105', 'protocol': 'TCP', 'port': 22, 'bytes': 2048, 'label': 'Intrusion'},
    {'id': 4, 'timestamp': '2026-02-07 10:15:35', 'src_ip': '172.16.0.88', 'dst_ip': '192.168.1.105', 'protocol': 'ICMP', 'port': 0, 'bytes': 512, 'label': 'DoS'},
    {'id': 5, 'timestamp': '2026-02-07 10:15:36', 'src_ip': '192.168.1.105', 'dst_ip': '1.1.1.1', 'protocol': 'TCP', 'port': 80, 'bytes': 896, 'label': 'Benign'},
    {'id': 6, 'timestamp': '2026-02-07 10:15:37', 'src_ip': '10.0.0.45', 'dst_ip': '192.168.1.105', 'protocol': 'TCP', 'port': 3389, 'bytes': 4096, 'label': 'Anomaly'},
]

@app.route('/')
@app.route('/dashboard')
def dashboard():
    analysis_results = session.get('analysis_results')
    analysis_loaded = bool(analysis_results)
    if analysis_loaded:
        # Карточки статистики — уже реальные
        stats = [
            {'label': 'Файл', 'value': analysis_results["filename"], 'change': '', 'trend': 'up', 'icon': 'activity',
             'color': 'blue'},
            {'label': 'Строк (событий)', 'value': str(analysis_results["rows"]), 'change': '', 'trend': 'up',
             'icon': 'activity', 'color': 'purple'},
            {'label': 'Обнаружено угроз', 'value': str(analysis_results["threats"]), 'change': '', 'trend': 'up',
             'icon': 'alert', 'color': 'red'},
            {'label': 'Модель', 'value': analysis_results["model_used"], 'change': '', 'trend': 'up',
             'icon': 'trending', 'color': 'green'},
        ]
        recent_analyses = session.get("recent_analyses", [])  # компактная история
        threat_distribution = analysis_results.get("threat_distribution", [])
    else:
        # Заглушки как раньше
        stats = [
            {'label': 'Всего анализов', 'value': '1,247', 'change': '+12.5%', 'trend': 'up', 'icon': 'activity',
             'color': 'blue'},
            {'label': 'Обнаружено атак', 'value': '342', 'change': '-8.2%', 'trend': 'down', 'icon': 'alert',
             'color': 'red'},
            {'label': 'Безопасный трафик', 'value': '89.3%', 'change': '+3.1%', 'trend': 'up', 'icon': 'check',
             'color': 'green'},
            {'label': 'Средняя точность', 'value': '96.4%', 'change': '+1.8%', 'trend': 'up', 'icon': 'trending',
             'color': 'purple'},
        ]
        recent_analyses = [
            {'id': 1, 'model': 'LightGBM', 'dataset': 'network_capture_020726.pcap', 'accuracy': '96.42%',
             'threats': 87, 'timestamp': '2026-02-07 09:23'},
            {'id': 2, 'model': 'XGBoost', 'dataset': 'traffic_data_020626.csv', 'accuracy': '95.18%', 'threats': 124,
             'timestamp': '2026-02-06 16:45'},
        ]
        threat_distribution = [
            {'type': 'DoS/DDoS', 'count': 145, 'percentage': 42, 'color': 'red'},
            {'type': 'Intrusion', 'count': 98, 'percentage': 29, 'color': 'orange'},
        ]
    return render_template(
        'dashboard.html',
        stats=stats,
        recent_analyses=recent_analyses,
        threat_distribution=threat_distribution,
        analysis_loaded=analysis_loaded,
        analysis_results=analysis_results,
    )

@app.route('/models')
def models():
    ml_models = [
        {
            'id': 'lightgbm',
            'name': 'LightGBM',
            'description': 'Gradient boosting framework с высокой производительностью для больших датасетов. Эффективен при обнаружении аномалий в сетевом трафике.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 7},
                {'label': 'Learning Rate', 'key': 'learning_rate', 'type': 'number', 'value': 0.1, 'step': 0.01},
            ]
        },
        {
            'id': 'xgboost',
            'name': 'XGBoost',
            'description': 'Оптимизированный gradient boosting алгоритм. Показывает высокую точность в задачах классификации сложных паттернов атак.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 6},
                {'label': 'Learning Rate', 'key': 'learning_rate', 'type': 'number', 'value': 0.3, 'step': 0.01},
            ]
        },
        {
            'id': 'random_forest',
            'name': 'Random Forest',
            'description': 'Ансамбль деревьев решений. Устойчив к переобучению, хорошо работает с высокоразмерными данными о сетевых соединениях.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 10},
                {'label': 'Критерий', 'key': 'criterion', 'type': 'select', 'value': 'gini', 'options': ['gini', 'entropy']},
            ]
        },
        {
            'id': 'isolation_forest',
            'name': 'Isolation Forest',
            'description': 'Unsupervised алгоритм для обнаружения аномалий. Идеален для выявления новых типов атак без предварительной разметки.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Contamination', 'key': 'contamination', 'type': 'number', 'value': 0.1, 'step': 0.01},
            ]
        },
    ]
    return render_template('models.html', models=ml_models)

@app.route('/data')
def data_upload():
    return render_template('data_upload.html', sample_data=SAMPLE_DATA, uploaded_file=None)

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


@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    # a) Принимаем имя файла (из формы) + fallback на last_uploaded_filename
    filename = request.form.get('filename') or session.get('last_uploaded_filename')
    if not filename:
        return jsonify({"error": "Не передано имя файла для анализа."}), 400

    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # b) Загружаем исходный CSV
    if not os.path.exists(filepath):
        return jsonify({"error": f"Файл не найден в uploads: {filename}"}), 404

    ext = os.path.splitext(filename)[1].lower()
    if ext != ".csv":
        return jsonify({"error": f"Формат {ext} не поддерживается для анализа (ожидается .csv)."}), 400

    try:
        loader = CSVDataLoader()
        raw_df = loader.load(filepath)

        # c) prepare
        processed_df = data_adapter.prepare(raw_df)

        # d) predict
        algo = request.form.get('algo', 'lightgbm')
        env = request.form.get('env', 'test')
        predictions = model_manager.predict(algo=algo, data=processed_df, env=env)

        # e) метрики
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

        # f) сохраняем результаты в session (компактно!)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        analysis_results = {
            "filename": filename,
            "timestamp": ts,
            "model_used": f"{algo}/{env}",
            "rows": total,
            "threats": threats,
            "class_counts": dict(counts),
            "threat_distribution": distribution,
            "predictions_sample": predictions[:200],  # НЕ всё, иначе cookie переполнится
        }
        session['analysis_results'] = analysis_results

        # (опционально) мини-история последних анализов (тоже компактно)
        history = session.get("recent_analyses", [])
        history.insert(0, {
            "model": algo,
            "dataset": filename,
            "accuracy": "-",           # если нет y_true, accuracy считать нельзя
            "threats": threats,
            "timestamp": ts.replace("T", " "),
        })
        session["recent_analyses"] = history[:10]

        # g) редирект на dashboard
        return redirect(url_for('dashboard'))

    except Exception as e:
        return jsonify({"error": f"Ошибка анализа: {str(e)}"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    original_filename = file.filename
    filename = secure_filename(original_filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['last_uploaded_filename'] = filename

    try:
        # 1. Выбор загрузчика в зависимости от расширения
        ext = os.path.splitext(filename)[1].lower()
        loader = None

        if ext == '.csv':
         loader = CSVDataLoader()
        # elif ext in ['.pcap', '.pcapng']:
        #     loader = PCAPDataLoader()
        else:
          return jsonify({'error': f'Формат {ext} не поддерживается'}), 400
        # 2. Загрузка сырых данных
        raw_df = loader.load(filepath)
        print(f"✅ raw_df загружен: {raw_df.shape}")# ← ОТЛАДКА
        print(f"raw_df: {raw_df[:5]}")

        # 3. Подготовка данных под модель
        processed_df = data_adapter.prepare(raw_df)
        print(f"✅ processed_df после подготовки: {processed_df.shape}")  # ← ОТЛАДКА
        print(f"✅ Колонки: {processed_df.columns.tolist()}")  # ← ОТЛАДКА
        print(f"✅ Первые 2 строки:\n{processed_df.head(2)}")  # ← ОТЛАДКА

        # 4. Предпросмотр для фронтенда
        sample_data = processed_df.head(6).to_dict(orient='records')
        columns = processed_df.columns.tolist()
        print(f"✅ sample_data (первые 2 записи): {sample_data[:2]}")  # ← ОТЛАДКА
        print(f"✅ Тип sample_data: {type(sample_data)}")
        print(f"✅ Длина sample_data: {len(sample_data)}")

        # Сохраняем обработанный DF во временное хранилище или сессию,
        # чтобы потом использовать его в /predict, если пользователь нажмет "Анализ"
        # Например, сохраняем в pickle во временную папку с уникальным ID
        import uuid
        session_id = str(uuid.uuid4())
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_processed.pkl")
        #processed_df.to_pickle(temp_path)
        raw_df.to_pickle(temp_path)

        uploaded_file_info = {
            'name': filename,
            'size': os.path.getsize(filepath),
            'session_id': session_id  # Передаем ID сессии для следующего шага
        }

        return render_template('data_upload.html',
                               sample_data=sample_data,
                               columns=columns,
                               uploaded_file=uploaded_file_info)
    except Exception as e:
        return jsonify({'error': f"Ошибка обработки данных: {str(e)}"}), 500


@app.route('/results')
def results():
    metrics = [
        {'label': 'Accuracy', 'value': '96.42%', 'icon': 'target', 'color': 'green'},
        {'label': 'F1-Score (Weighted)', 'value': '95.87%', 'icon': 'trending-up', 'color': 'blue'},
        {'label': 'ROC-AUC', 'value': '0.9823', 'icon': 'zap', 'color': 'purple'},
        {'label': 'Latency', 'value': '12.4 ms', 'icon': 'clock', 'color': 'orange'},
    ]
    
    class_metrics = [
        {'class': 'DoS/DDoS', 'precision': '98.2%', 'recall': '97.5%', 'f1_score': '97.8%', 'support': 2453, 'type': 'attack'},
        {'class': 'Intrusion', 'precision': '94.7%', 'recall': '93.1%', 'f1_score': '93.9%', 'support': 1847, 'type': 'attack'},
        {'class': 'Anomaly', 'precision': '91.3%', 'recall': '89.8%', 'f1_score': '90.5%', 'support': 892, 'type': 'attack'},
        {'class': 'Benign', 'precision': '99.1%', 'recall': '99.4%', 'f1_score': '99.2%', 'support': 8763, 'type': 'benign'},
    ]
    
    return render_template('results.html', metrics=metrics, class_metrics=class_metrics)

@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Проверяем, пришёл ли JSON
    if request.is_json:
        data = request.get_json()
        selected_model_id = data.get('algo', 'lightgbm')
        env = data.get('env', 'test')
        # params = data.get('params', {})
    else:
        # Если не JSON — читаем как form-data (старый способ)
        selected_model_id = request.form.get('selected_model', 'lightgbm')
        env = request.form.get('env', 'test')

    # Проверим, поддерживает ли ModelManager эту модель
    if selected_model_id not in model_manager.file_map:
        return jsonify({"error": f"Модель '{selected_model_id}' не поддерживается."}), 400

    try:
        # Загрузим bundle, чтобы получить feature_names
       # bundle = model_manager._get_or_load_bundle(algo=selected_model_id, env=env)
        #feature_names = bundle.feature_names

        # Создаём DataFrame с нужными колонками, заполненный нулями
       # dummy_data = pd.DataFrame([0.0] * len(feature_names)).T
        #dummy_data = np.zeros((1, 100), dtype=np.float32)
        # Для теста: используем реальный DataFrame с правильными колонками
        # Пример: возьмём колонки из вашего X_train (если есть)
        # Если нет — создадим заглушку с именами, как при обучении
        # Загружаем информацию о признаках
        feature_info = joblib.load('models/lightgbm/test/feature_info.pkl')
        all_feature_names = feature_info['numeric_features'] + feature_info['categorical_features']

        # Создаём dummy DataFrame
        dummy_df = pd.DataFrame([0.0] * len(all_feature_names)).T
        print(dummy_df)
        dummy_df.columns = all_feature_names

        predictions = model_manager.predict("lightgbm", dummy_df)
        predicted_class = predictions[0] if predictions else "Unknown"

        # Вызываем модель
        #predictions = model_manager.predict(algo=selected_model_id, data=dummy_data, env=env)
        #predictions = model_manager.predict("lightgbm", dummy_data)

        #Для теста
        '''bundle = model_manager._get_or_load_bundle(algo=selected_model_id, env=env)
        X = pd.DataFrame([0.0] * len(bundle.feature_names)).T
        X.columns = bundle.feature_names
        pred = bundle.model.predict(X.values, predict_disable_shape_check=True)
        predictions = [str(x) for x in pred]'''
        #Возвращаем результат
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "predicted_class": predicted_class,
            "count": len(predictions),
            "model_used": f"{selected_model_id}/{env}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
