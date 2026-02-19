# modules_site/results/routes.py
from flask import render_template
from . import bp  # Импортируем свой Blueprint

@bp.route('/')
def results():
    """Роут для страницы /results"""
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