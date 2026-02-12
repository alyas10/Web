from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from model_manager.model_manager import ModelManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Инициализируем менеджер моделей
model_manager = ModelManager(models_root="models")

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
def dashboard():
    stats = [
        {'label': 'Всего анализов', 'value': '1,247', 'change': '+12.5%', 'trend': 'up', 'icon': 'activity', 'color': 'blue'},
        {'label': 'Обнаружено атак', 'value': '342', 'change': '-8.2%', 'trend': 'down', 'icon': 'alert', 'color': 'red'},
        {'label': 'Безопасный трафик', 'value': '89.3%', 'change': '+3.1%', 'trend': 'up', 'icon': 'check', 'color': 'green'},
        {'label': 'Средняя точность', 'value': '96.4%', 'change': '+1.8%', 'trend': 'up', 'icon': 'trending', 'color': 'purple'},
    ]
    
    recent_analyses = [
        {'id': 1, 'model': 'LightGBM', 'dataset': 'network_capture_020726.pcap', 'accuracy': '96.42%', 'threats': 87, 'timestamp': '2026-02-07 09:23'},
        {'id': 2, 'model': 'XGBoost', 'dataset': 'traffic_data_020626.csv', 'accuracy': '95.18%', 'threats': 124, 'timestamp': '2026-02-06 16:45'},
        {'id': 3, 'model': 'Random Forest', 'dataset': 'network_scan_020526.pcap', 'accuracy': '94.73%', 'threats': 56, 'timestamp': '2026-02-05 14:12'},
        {'id': 4, 'model': 'Isolation Forest', 'dataset': 'anomaly_detection_020426.csv', 'accuracy': '91.86%', 'threats': 203, 'timestamp': '2026-02-04 11:38'},
    ]
    
    threat_distribution = [
        {'type': 'DoS/DDoS', 'count': 145, 'percentage': 42, 'color': 'red'},
        {'type': 'Intrusion', 'count': 98, 'percentage': 29, 'color': 'orange'},
        {'type': 'Anomaly', 'count': 68, 'percentage': 20, 'color': 'yellow'},
        {'type': 'Other', 'count': 31, 'percentage': 9, 'color': 'gray'},
    ]
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_analyses=recent_analyses,
                         threat_distribution=threat_distribution)

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    uploaded_file = {
        'name': filename,
        'size': os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    }
    
    return render_template('data_upload.html', sample_data=SAMPLE_DATA, uploaded_file=uploaded_file)

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
    """
    Выполняет предсказание для выбранной модели.
    Пока что без загрузки данных – вызываем модель с "пустыми" признаками.
    """
    selected_model_id = request.form.get('selected_model', 'lightgbm')
    # env – пока всегда test (позже можно добавить выбор)
    env = 'test'

    # Проверим, поддерживает ли ModelManager эту модель
    if selected_model_id not in model_manager.file_map:
        return jsonify({"error": f"Модель '{selected_model_id}' не поддерживается."}), 400

    try:
        # ВНИМАНИЕ: на данном этапе у нас нет данных, чтобы передать в predict()
        # Пока что вызовем с "пустым" DataFrame (только нужные колонки)
        # Заглушка: пустой DataFrame с нужными колонками (например, 60 признаков)
        # В реальности: это будет результат DataLoaderFactory.process_file()
        import pandas as pd
        dummy_data = pd.DataFrame({f"feature_{i}": [0.0] for i in range(60)})

        # Вызываем модель
        predictions = model_manager.predict(algo=selected_model_id, data=dummy_data, env=env)

        # Возвращаем результат
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "count": len(predictions),
            "model_used": f"{selected_model_id}/{env}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
