# modules_site/dashboard/routes.py
from flask import render_template, session, jsonify
from . import bp  # Импортируем Blueprint из текущего пакета
import json
from pathlib import Path

# Глобальные объекты, которые использовались в app.py
# Они будут доступны через app (см. app.py)
# model_manager, data_adapter, visualizer, feature_info, SAMPLE_DATA

# SAMPLE_DATA
SAMPLE_DATA = [
    {'id': 1, 'timestamp': '2026-02-07 10:15:32', 'src_ip': '192.168.1.105', 'dst_ip': '8.8.8.8', 'protocol': 'TCP', 'port': 443, 'bytes': 1452, 'label': 'Benign'},
    {'id': 2, 'timestamp': '2026-02-07 10:15:33', 'src_ip': '192.168.1.105', 'dst_ip': '203.0.113.45', 'protocol': 'UDP', 'port': 53, 'bytes': 128, 'label': 'Benign'},
    {'id': 3, 'timestamp': '2026-02-07 10:15:34', 'src_ip': '10.0.0.23', 'dst_ip': '192.168.1.105', 'protocol': 'TCP', 'port': 22, 'bytes': 2048, 'label': 'Intrusion'},
    {'id': 4, 'timestamp': '2026-02-07 10:15:35', 'src_ip': '172.16.0.88', 'dst_ip': '192.168.1.105', 'protocol': 'ICMP', 'port': 0, 'bytes': 512, 'label': 'DoS'},
    {'id': 5, 'timestamp': '2026-02-07 10:15:36', 'src_ip': '192.168.1.105', 'dst_ip': '1.1.1.1', 'protocol': 'TCP', 'port': 80, 'bytes': 896, 'label': 'Benign'},
    {'id': 6, 'timestamp': '2026-02-07 10:15:37', 'src_ip': '10.0.0.45', 'dst_ip': '192.168.1.105', 'protocol': 'TCP', 'port': 3389, 'bytes': 4096, 'label': 'Anomaly'},
]

# Функция для определения цвета метки (была в app.py)
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

@bp.route('/')
@bp.route('/dashboard')
def dashboard():
    # Модели для отображения в разделе "Активные модели" с описанием
    active_models = [
        {'name': 'LightGBM',
         'description': 'Gradient boosting framework с высокой производительностью. Эффективен при обнаружении аномалий в сетевом трафике.'},
        {'name': 'XGBoost',
         'description': 'Оптимизированный gradient boosting алгоритм. Показывает высокую точность в задачах классификации сложных паттернов атак.'},
        {'name': 'Random Forest',
         'description': 'Ансамбль деревьев решений. Устойчив к переобучению, хорошо работает с высокоразмерными данными.'},
        {'name': 'Isolation Forest',
         'description': 'Unsupervised алгоритм для обнаружения аномалий. Идеален для выявления новых типов атак без разметки.'},
    ]
    analysis_results = session.get('analysis_results')
    analysis_loaded = bool(analysis_results)
    if analysis_loaded:
        threat_types = analysis_results.get("threat_types",
                                            len(analysis_results.get("threat_distribution", [])))
        stats = [
            {'label': 'Строк (событий)', 'value': str(analysis_results["rows"]),
             'change': '', 'trend': 'up', 'icon': 'activity', 'color': 'blue'},
            {'label': 'Событий угрозы', 'value': str(analysis_results["threats"]),
             'change': '', 'trend': 'up', 'icon': 'alert', 'color': 'red'},
            {'label': 'Типов угроз', 'value': str(threat_types),
             'change': '', 'trend': 'up', 'icon': 'check', 'color': 'orange'},
            {'label': 'Модель', 'value': analysis_results["model_used"],
             'change': '', 'trend': 'up', 'icon': 'trending', 'color': 'green'},
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
        active_models=active_models,
    )

@bp.route('/api/dashboard-data')
def api_dashboard_data():
    """API endpoint для получения данных интерактивных дашбордов"""
    analysis_results = session.get('analysis_results')

    if analysis_results:
        # Реальные данные из последнего анализа
        total_events = analysis_results.get('rows', 0)
        threats = analysis_results.get('threats', 0)
        class_counts = analysis_results.get('class_counts', {})
        # Берем Benign из class_counts если доступно, иначе вычисляем
        benign_count = class_counts.get('Benign', class_counts.get('benign', 0))
        safe_traffic = benign_count if benign_count > 0 else max(0, total_events - threats)

        # Распределение классов (если есть в результатах)
        if class_counts:
            class_distribution = dict(class_counts)
        else:
            class_dist = analysis_results.get('threat_distribution', [])
            class_distribution = {'Benign': safe_traffic}
            for item in class_dist:
                class_distribution[item.get('type', 'Unknown')] = item.get('count', 0)

        # История анализов из session
        recent_analyses = session.get('recent_analyses', [])
        analysis_history = []
        for analysis in recent_analyses[-5:]:  # Последние 5
            analysis_history.append({
                'date': analysis.get('timestamp', '')[:10],
                'threats': analysis.get('threats', 0)
            })

        # Если истории нет, добавляем текущий анализ
        if not analysis_history and analysis_results:
            analysis_history.append({
                'date': analysis_results.get('timestamp', '')[:10],
                'threats': threats
            })

        # Топ угроз
        top_threats = []
        for item in class_dist:
            if item.get('type', '').lower() != 'benign':
                top_threats.append({
                    'type': item.get('type', 'Unknown'),
                    'count': item.get('count', 0)
                })
        top_threats.sort(key=lambda x: x['count'], reverse=True)

        # Метрики моделей (из файлов metrics.json)
        model_metrics = _get_model_metrics()

        return jsonify({
            'total_events': total_events,
            'threats_detected': threats,
            'safe_traffic': safe_traffic,
            'model_used': analysis_results.get('model_used', 'Unknown'),
            'is_demo': False,
            'class_distribution': class_distribution,
            'model_metrics': model_metrics,
            'analysis_history': analysis_history,
            'top_threats': top_threats[:5]  # Топ 5
        })

    # Демо-данные если нет результатов анализа
    return jsonify(_get_demo_dashboard_data())


def _get_model_metrics():
    """Получение метрик моделей из сохраненных файлов"""
    models_data = []
    model_configs = [
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('random_forest', 'Random Forest'),
        ('isolation_forest', 'Isolation Forest')
    ]

    for model_id, model_name in model_configs:
        try:
            # Путь к метрикам модели
            metrics_path = Path(f'pipeline/{model_id}/test/metrics.json')
            if metrics_path.exists():
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                accuracy = metrics.get('accuracy', metrics.get('test_accuracy', 0))
                f1 = metrics.get('f1_weighted',
                                metrics.get('f1_macro',
                                           metrics.get('f1_score', 0)))

                models_data.append({
                    'name': model_name,
                    'accuracy': accuracy,
                    'f1': f1
                })
        except Exception as e:
            # Если не удалось загрузить, используем значения по умолчанию
            pass

    # Если ничего не загрузилось, возвращаем дефолтные значения
    if not models_data:
        models_data = [
            {'name': 'LightGBM', 'accuracy': 0.964, 'f1': 0.952},
            {'name': 'XGBoost', 'accuracy': 0.951, 'f1': 0.943},
            {'name': 'Random Forest', 'accuracy': 0.938, 'f1': 0.921},
            {'name': 'Isolation Forest', 'accuracy': 0.892, 'f1': 0.876}
        ]

    return models_data


def _get_demo_dashboard_data():
    """Демо-данные для дашборда"""
    return {
        'total_events': 15847,
        'threats_detected': 342,
        'safe_traffic': 15505,
        'model_used': 'LightGBM',
        'is_demo': True,
        'class_distribution': {
            'Benign': 12450,
            'DoS/DDoS': 1820,
            'Intrusion': 987,
            'Anomaly': 456,
            'Port Scan': 134
        },
        'model_metrics': [
            {'name': 'LightGBM', 'accuracy': 0.964, 'f1': 0.952},
            {'name': 'XGBoost', 'accuracy': 0.951, 'f1': 0.943},
            {'name': 'Random Forest', 'accuracy': 0.938, 'f1': 0.921},
            {'name': 'Isolation Forest', 'accuracy': 0.892, 'f1': 0.876}
        ],
        'analysis_history': [
            {'date': '2026-02-07', 'threats': 87},
            {'date': '2026-02-06', 'threats': 124},
            {'date': '2026-02-05', 'threats': 56},
            {'date': '2026-02-04', 'threats': 93},
            {'date': '2026-02-03', 'threats': 71}
        ],
        'top_threats': [
            {'type': 'DoS/DDoS', 'count': 1820},
            {'type': 'Intrusion', 'count': 987},
            {'type': 'Anomaly', 'count': 456},
            {'type': 'Port Scan', 'count': 134}
        ]
    }
