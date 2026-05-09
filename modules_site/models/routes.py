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
import lightgbm as lgb
import xgboost as xgb
from sklearn import set_config
import matplotlib

# Регистрируем класс в глобальной области видимости для корректной загрузки пайплайна
from model_manager.model_utils import NumericFeatureSelector
import sys
sys.modules['__main__'].NumericFeatureSelector = NumericFeatureSelector

set_config(display='diagram')
matplotlib.use('agg')

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


@contextmanager
def plt_context():
    """Контекст для безопасной работы с matplotlib"""
    try:
        yield
    finally:
        plt.close('all')  # Гарантированно закрываем все фигуры


def _plot_to_base64(dpi=120):
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
    Генерирует визуализацию для указанной модели (LightGBM/XGBoost).
    Работает как с Pipeline, так и с прямыми моделями из joblib.
    """
    try:
        # Загружаем бандл через ModelManager
        bundle = current_app.model_manager._get_or_load_bundle(model_id, 'test')
        model_obj = bundle.pipeline  # Это может быть Pipeline или прямой классификатор

        # === Определяем тип объекта и извлекаем классификатор ===
        if hasattr(model_obj, 'named_steps') and 'classifier' in model_obj.named_steps:
            # Это полноценный sklearn Pipeline
            pipeline = model_obj
            classifier = pipeline.named_steps['classifier']
            is_pipeline = True
        else:
            # Это прямой классификатор (XGBClassifier, LGBMClassifier и т.д.)
            pipeline = None
            classifier = model_obj
            is_pipeline = False

        # === Получаем имена признаков ===
        feature_names = None

        # 1. Пробуем получить из самого классификатора
        if hasattr(classifier, 'feature_names_in_'):
            feature_names = classifier.feature_names_in_

        # 2. Если нет — пробуем из pipeline (если он есть)
        if feature_names is None and is_pipeline and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    pass

        # 3. Если всё ещё нет — заглушка
        if feature_names is None:
            # Пробуем получить из current_app
            feature_names = getattr(current_app, 'REQUIRED_FEATURES',
                                    [f'Feature_{i}' for i in range(100)])

        # === ГРАФИК 1: Важность признаков (универсальный) ===
        importance_img = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            n_features = min(10, len(importances))
            indices = np.argsort(importances)[-n_features:][::-1]

            plt.figure(figsize=(10, 6), facecolor='#1f2937')
            plt.barh(range(n_features), importances[indices], color='#2E86AB')

            # Подписи осей
            labels = []
            for idx in indices:
                if idx < len(feature_names):
                    name = feature_names[idx]
                    # Обрезаем слишком длинные имена
                    if len(name) > 25:
                        name = name[:22] + '...'
                    labels.append(name)
                else:
                    labels.append(f'Feature_{idx}')

            plt.yticks(range(n_features), labels, fontsize=7)
            plt.xlabel('Важность (Gain)', color='#9ca3af', fontsize=9)
            plt.title('Важность признаков', color='white', fontsize=12)
            plt.tick_params(colors='#9ca3af', labelsize=8)
            plt.tight_layout()
            importance_img = _plot_to_base64()
            plt.close()

        # === ГРАФИК 2: Дерево решений (если поддерживается) ===
        tree_img = None
        try:
            # === LightGBM ===
            if hasattr(classifier, 'booster_'):
                import lightgbm as lgb
                booster = classifier.booster_
                if hasattr(classifier, 'num_leaves') and classifier.num_leaves <= 31:
                    plt.figure(figsize=(20, 12), facecolor='#1f2937')
                    lgb.plot_tree(booster, tree_index=0, figsize=(12, 8))
                    plt.title('Дерево решений №1', color='white', fontsize=12)
                    plt.axis('off')
                    tree_img = _plot_to_base64()
                    plt.close()

            # === XGBoost ===
            elif hasattr(classifier, 'get_booster'):
                import xgboost as xgb
                booster = classifier.get_booster()
                # Проверяем количество деревьев перед отрисовкой
                if hasattr(booster, 'num_boosted_rounds'):
                    n_trees = booster.num_boosted_rounds()
                else:
                    n_trees = classifier.n_estimators if hasattr(classifier, 'n_estimators') else 1

                print(f"[DEBUG] XGBoost: n_trees={n_trees}, booster type={type(booster)}")
                try:
                    plt.figure(figsize=(24, 16), facecolor='#1f2937', dpi=150)
                    xgb.plot_tree(
                        booster,
                        num_trees=0,  # ← Только первое дерево
                        rankdir='LR',  # ← Горизонтальная ориентация
                        # Параметры для узлов
                        condition_node_params={
                            'shape': 'box',
                            'style': 'filled,rounded',
                            'fillcolor': '#78bceb'
                        },
                        # Параметры для листьев
                        leaf_node_params={
                            'shape': 'box',
                            'style': 'filled',
                            'fillcolor': '#e48038'
                        }
                    )
                    plt.title(f'Дерево решений №1 (из {booster.num_boosted_rounds()})',
                              color='white', fontsize=11)
                    plt.axis('off')
                    tree_img = _plot_to_base64()
                    plt.close()
                except Exception as tree_err:
                    print(f"⚠️ Дерево не отрисовано для {model_id}: {tree_err}")
                    tree_img = None

        except Exception as tree_err:
            print(f"⚠️ Дерево не отрисовано для {model_id}: {tree_err}")
            tree_img = None

        # === Pipeline HTML (только если это настоящий Pipeline) ===
        pipeline_html = None
        if is_pipeline and hasattr(pipeline, '_repr_html_'):
            pipeline_html = pipeline._repr_html_()

        return {
            'success': True,
            'importance': importance_img,
            'pipeline': pipeline_html,
            'pipeline_type': 'html' if pipeline_html else None,
            'tree': tree_img,
            'info': {
                'model_type': type(classifier).__name__,
                'features': len(importances) if 'importances' in locals() else 'N/A',
                'is_pipeline': is_pipeline
            }
        }

    except Exception as e:
        print(f"❌ Ошибка визуализации {model_id}: {e}")
        import traceback
        traceback.print_exc()
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
            'has_viz': True
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
            'has_viz': True
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
    from flask import session, redirect, url_for
    from datetime import datetime

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

        # Создаём dummy DataFrame с одним нулевым значением для каждого признака
        dummy_data = {col: [0.0] for col in all_feature_names}
        dummy_df = pd.DataFrame(dummy_data)

        predictions = model_manager.predict(selected_model_id, dummy_df, env)
        predicted_class = predictions[0] if predictions else "Unknown"

        # Подсчитываем угрозы
        threats_count = sum(1 for p in predictions if p != 'Benign')

        # Формируем распределение угроз
        threat_dist = {}
        for pred in predictions:
            if pred != 'Benign':
                threat_dist[pred] = threat_dist.get(pred, 0) + 1

        threat_distribution = [
            {'type': k, 'count': v, 'percentage': round(v / len(predictions) * 100) if predictions else 0,
             'color': 'red' if 'DoS' in k else 'orange' if 'Intrusion' in k else 'yellow'}
            for k, v in threat_dist.items()
        ] if threat_dist else [{'type': 'Нет угроз', 'count': 0, 'percentage': 0, 'color': 'green'}]

        # === СОХРАНЯЕМ РЕЗУЛЬТАТЫ В SESSION ===
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        analysis_results = {
            "filename": f"analysis_{selected_model_id}_{timestamp.replace(':', '-')}.csv",
            "rows": len(predictions),
            "threats": threats_count,
            "model_used": f"{selected_model_id}/{env}",
            "timestamp": timestamp,
            "threat_distribution": threat_distribution,
            "accuracy": "96.42%"  # Можно рассчитать реально если есть тестовые данные
        }

        # Сохраняем в session
        session['analysis_results'] = analysis_results

        # Добавляем в историю последних анализов
        recent_analyses = session.get('recent_analyses', [])
        recent_analyses.insert(0, {
            'id': len(recent_analyses) + 1,
            'model': selected_model_id.title().replace('_', ' '),
            'dataset': analysis_results['filename'],
            'accuracy': analysis_results['accuracy'],
            'threats': threats_count,
            'timestamp': timestamp
        })
        # Храним только последние 5
        session['recent_analyses'] = recent_analyses[:5]

        # Возвращаем результат
        return jsonify({
            "status": "success",
            "predictions": predictions,
            "predicted_class": predicted_class,
            "count": len(predictions),
            "model_used": f"{selected_model_id}/{env}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_pipeline_html(pipeline):
    """
    Возвращает готовый HTML от sklearn (как в Jupyter).
    """
    # sklearn сам генерирует HTML через _repr_html_()
    if hasattr(pipeline, '_repr_html_'):
        return pipeline._repr_html_()
    return None


def _plot_pipeline_structure(pipeline, feature_info=None, dpi=120):
    """
    Рисует структуру sklearn Pipeline как блок-схему через matplotlib.
    Возвращает base64-строку изображения.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Настройки стиля
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='#1f2937')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(pipeline.steps) * 2 + 1)
    ax.axis('off')

    # Цвета для разных типов шагов
    colors = {
        'preprocessor': '#3B82F6',  # синий
        'feature_selector': '#10B981',  # зелёный
        'classifier': '#F59E0B',  # оранжевый
        'default': '#6B7280'  # серый
    }

    # Рисуем каждый шаг пайплайна
    for i, (step_name, step_obj) in enumerate(pipeline.steps):
        y_pos = len(pipeline.steps) * 2 - i * 2

        # Определяем тип шага для цвета
        step_type = 'default'
        if 'preprocess' in step_name.lower() or 'column' in type(step_obj).__name__.lower():
            step_type = 'preprocessor'
        elif 'select' in step_name.lower() or 'feature' in step_name.lower():
            step_type = 'feature_selector'
        elif 'classif' in step_name.lower() or 'regress' in step_name.lower():
            step_type = 'classifier'

        # Рисуем прямоугольник шага
        rect = patches.Rectangle(
            (1, y_pos - 0.4), 8, 0.8,
            linewidth=2, edgecolor='white', facecolor=colors.get(step_type, colors['default']),
            alpha=0.9)
        ax.add_patch(rect)

        # Текст: название шага
        ax.text(5, y_pos, f'{step_name}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Текст: тип объекта (мелким шрифтом)
        obj_name = type(step_obj).__name__
        if hasattr(step_obj, '__class__'):
            obj_name = f"{step_obj.__class__.__module__.split('.')[-1]}.{obj_name}"
        ax.text(5, y_pos - 0.6, obj_name, ha='center', va='top',
                fontsize=8, color='#9ca3af', style='italic')

        # Стрелка между шагами (кроме последнего)
        if i < len(pipeline.steps) - 1:
            ax.annotate('', xy=(5, y_pos - 1.2), xytext=(5, y_pos - 0.6),
                        arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.5))

    # Заголовок схемы
    ax.text(5, len(pipeline.steps) * 2 + 0.5, ' Структура ML Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Легенда с типами шагов
    legend_y = -0.5
    for label, color in colors.items():
        if label != 'default':
            ax.add_patch(patches.Rectangle((0.5, legend_y), 0.3, 0.3,
                                           facecolor=color, edgecolor='white', alpha=0.9))
            ax.text(1, legend_y + 0.15, label.replace('_', ' ').title(),
                    fontsize=8, color='#9ca3af', va='center')
            legend_y -= 0.4

    # Информация о признаках (если передана)
    if feature_info:
        info_text = f"Признаков: {feature_info.get('n_features_in', 'N/A')}\n"
        if 'numeric_features' in feature_info:
            info_text += f"• Числовые: {len(feature_info['numeric_features'])}\n"
        if 'categorical_features' in feature_info:
            info_text += f"• Категориальные: {len(feature_info['categorical_features'])}"
        ax.text(9.8, 0.3, info_text, ha='right', va='bottom',
                fontsize=8, color='#6B7280', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#111827', edgecolor='#374151', alpha=0.8))

    plt.tight_layout()

    # Конвертируем в base64
    return _plot_to_base64(dpi=dpi)
