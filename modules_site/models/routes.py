# modules_site/pipeline/routes.py
from flask import render_template, request, jsonify
from flask import current_app  # <-- Импортируем current_app для доступа к объектам из app
from . import bp  # Импортируем свой Blueprint
import pandas as pd
import numpy as np
import joblib

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
        if feature_info_path: # Используем загруженный ранее feature_info
            all_feature_names = current_app.REQUIRED_FEATURES # или feature_info['numeric_features'] + feature_info['categorical_features']

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