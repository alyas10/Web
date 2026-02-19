# modules_site/dashboard/routes.py
from flask import render_template, session
from . import bp  # Импортируем Blueprint из текущего пакета

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
    analysis_results = session.get('analysis_results')
    analysis_loaded = bool(analysis_results)
    if analysis_loaded:
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