# modules_site/pipeline/routes.py
from flask import render_template, request, jsonify
from flask import current_app  # <-- Импортируем current_app для доступа к объектам из app
from . import bp  # Импортируем свой Blueprint
import pandas as pd
import numpy as np
import joblib
from contextlib import contextmanager
import io
import os
import base64
import matplotlib.pyplot as plt
import lightgbm as lgb  # ← ДОБАВЛЕНО: нужен для plot_importance и plot_tree

import matplotlib
matplotlib.use('agg')

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

@contextmanager
def plt_context():
    """Контекст для безопасной работы с matplotlib"""
    try:
        yield
    finally:
        plt.close('all')  # Гарантированно закрываем все фигуры


def _plot_to_base64(dpi=100):
    """
    Конвертирует текущий matplotlib график в base64-строку.

    Почему base64: JSON не умеет передавать бинарные данные (картинки).
    Base64 превращает байты картинки в текст, который можно вставить в <img src="...">
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, facecolor='#1f2937')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()  # Обязательно закрываем, чтобы не было утечки памяти
    return img_base64


def _get_model_viz(model_id):
    """
    Генерирует визуализацию для указанной модели.
    Берёт пайплайн из current_app.pipeline (загружен в app.py при старте)
    """
    try:
        # ← ИСПРАВЛЕНО: берём пайплайн из current_app вместо загрузки из файла
        pipeline = current_app.pipeline

        if pipeline is None:
            return {'success': False, 'error': 'Пайплайн не загружен при старте приложения'}

        # Извлекаем обученную LightGBM модель из пайплайна
        lgb_model = pipeline.named_steps['classifier']
        booster = lgb_model.booster_

        # === ГРАФИК 1: Важность признаков ===
        plt.figure(figsize=(10, 6), facecolor='#1f2937')
        lgb.plot_importance(booster, max_num_features=10, height=0.5, color='#2E86AB')
        plt.title('Важность признаков', color='white', fontsize=12)
        plt.tick_params(colors='#9ca3af')  # Цвет подписей осей
        importance_img = _plot_to_base64()

        # === ГРАФИК 2: Дерево решений (только если небольшое) ===
        tree_img = None
        if lgb_model.num_leaves <= 31:  # Большие деревья не рисуем — будет каша
            plt.figure(figsize=(12, 8), facecolor='#1f2937')
            lgb.plot_tree(booster, tree_index=0, figsize=(12, 8))
            plt.title('Дерево решений №1', color='white', fontsize=12)
            plt.axis('off')
            tree_img = _plot_to_base64()

        return {
            'success': True,
            'importance': importance_img,
            'tree': tree_img,
            'info': {
                'trees': booster.num_trees(),
                'features': booster.num_feature(),
                'leaves': lgb_model.num_leaves
            }
        }  # ← ИСПРАВЛЕНО: добавлена закрывающая скобка

    except Exception as e:
        return {'success': False, 'error': str(e)}


@bp.route('/')
def models():
    """Роут для страницы /pipeline"""
    ml_models = [
        {
            'id': 'lightgbm',
            'name': 'LightGBM',
            'description': 'Gradient boosting framework с высокой производительностью для больших датасетов. Эффективен при обнаружении аномалий в сетевом трафике.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 7},
                {'label': 'Learning Rate', 'key': 'learning_rate', 'type': 'number', 'value': 0.1, 'step': 0.01},
            ],
            'has_viz': True  # ← ДОБАВЛЕНО: флаг для визуализации
        },
        {
            'id': 'xgboost',
            'name': 'XGBoost',
            'description': 'Оптимизированный gradient boosting алгоритм. Показывает высокую точность в задачах классификации сложных паттернов атак.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 6},
                {'label': 'Learning Rate', 'key': 'learning_rate', 'type': 'number', 'value': 0.3, 'step': 0.01},
            ],
            'has_viz': False  # Пока нет визуализации для других моделей
        },
        {
            'id': 'random_forest',
            'name': 'Random Forest',
            'description': 'Ансамбль деревьев решений. Устойчив к переобучению, хорошо работает с высокоразмерными данными о сетевых соединениях.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Глубина дерева', 'key': 'max_depth', 'type': 'number', 'value': 10},
                {'label': 'Критерий', 'key': 'criterion', 'type': 'select', 'value': 'gini',
                 'options': ['gini', 'entropy']},
            ],
            'has_viz': False
        },
        {
            'id': 'isolation_forest',
            'name': 'Isolation Forest',
            'description': 'Unsupervised алгоритм для обнаружения аномалий. Идеален для выявления новых типов атак без предварительной разметки.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 100},
                {'label': 'Contamination', 'key': 'contamination', 'type': 'number', 'value': 0.1, 'step': 0.01},
            ],
            'has_viz': False
        },
    ]
    return render_template('models.html', models=ml_models)


# ← ДОБАВЛЕНО: API роут для визуализации
@bp.route('/api/viz/<model_id>')
def model_viz_api(model_id):
    """API endpoint для получения визуализации модели"""
    result = _get_model_viz(model_id)
    return jsonify(result)


@bp.route('/predict', methods=['POST'])
def predict():
    """Роут для предсказания POST /pipeline/predict"""
    model_manager = current_app.model_manager
    feature_info = current_app.feature_info

    # Проверяем, пришёл ли JSON
    if request.is_json:
        data = request.get_json()
        selected_model_id = data.get('algo', 'lightgbm')
        env = data.get('env', 'test')
    else:
        # Если не JSON — читаем как form-data (старый способ)
        selected_model_id = request.form.get('selected_model', 'lightgbm')
        env = request.form.get('env', 'test')

    # Проверим, поддерживает ли ModelManager эту модель
    if selected_model_id not in model_manager.file_map:
        return jsonify({"error": f"Модель '{selected_model_id}' не поддерживается."}), 400

    try:
        # Загружаем информацию о признаках (если нужно для создания заглушки)
        # all_feature_names = feature_info['numeric_features'] + feature_info['categorical_features']

        # Для теста: используем реальный DataFrame с правильными колонками
        # Загружаем информацию о признаках
        feature_info_path = 'pipeline/lightgbm/test/feature_info.pkl'
        if feature_info_path:  # Используем загруженный ранее feature_info
            all_feature_names = current_app.REQUIRED_FEATURES  # или feature_info['numeric_features'] + feature_info['categorical_features']

        # Создаём dummy DataFrame
        dummy_df = pd.DataFrame([0.0] * len(all_feature_names)).T
        dummy_df.columns = all_feature_names

        predictions = model_manager.predict(selected_model_id, dummy_df, env)
        predicted_class = predictions[0] if predictions else "Unknown"

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "predicted_class": predicted_class,
            "count": len(predictions),
            "model_used": f"{selected_model_id}/{env}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500