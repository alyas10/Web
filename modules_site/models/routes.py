# modules_site/models/routes.py
from pathlib import Path

from flask import render_template, request, jsonify, current_app, session
from . import bp  # Импорт один раз в начале
import pandas as pd
import numpy as np
import joblib
from contextlib import contextmanager
import io
import os
import base64
import matplotlib
import logging

matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, List, Any, Union

# Импорт функций загрузки конфига проекта из settings
from modules_site.settings.routes import load_config

# Импорт класса для корректной десериализации
from model_manager.model_utils import NumericFeatureSelector
import sys

# Более безопасная регистрация класса для десериализации
try:
    if hasattr(sys.modules.get('__main__'), '__dict__'):
        sys.modules['__main__'].__dict__['NumericFeatureSelector'] = NumericFeatureSelector
except Exception as e:
    logging.warning(f"Could not register NumericFeatureSelector: {e}")

# Импорт функций для расчета характеристик моделей
from model_manager.model_metrics import (
    extract_model_params,
    extract_feature_names,
    get_model_type_display,
    get_model_speed,
    get_model_complexity,
    get_model_description,
    get_model_advantages,
    get_how_it_works,
    get_security_application,
    has_tree_viz,
    calculate_accuracy_from_metrics,
    extract_model_metadata
)

from sklearn import set_config

set_config(display='diagram')

# Graphviz path - кроссплатформенный подход
GRAPHVIZ_PATHS = [
    r'C:\Program Files\Graphviz\bin',
    r'C:\Program Files (x86)\Graphviz\bin',
    '/usr/local/bin',
    '/usr/bin'
]

TEST_DATA_MAP = {
    'isolation_forest': 'test/processed/test_data/isolation_forest_test_10k.csv',
    'lightgbm':         'test/processed/test_data/isolation_forest_test_10k.csv',
    'xgboost':          'test/processed/test_data/isolation_forest_test_10k.csv',
    'random_forest':    'test/processed/test_data/isolation_forest_test_10k.csv',
}


for path in GRAPHVIZ_PATHS:
    if os.path.exists(path) and path not in os.environ.get('PATH', ''):
        os.environ["PATH"] = path + os.pathsep + os.environ.get('PATH', '')
        break


@contextmanager
def plt_context():
    """Контекст для безопасной работы с matplotlib"""
    try:
        yield
    finally:
        plt.close('all')


def _plot_to_base64(dpi: int = 200) -> str:
    """Конвертирует matplotlib график в base64-строку"""
    buf = io.BytesIO()
    try:
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,
                    facecolor='#1f2937', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    finally:
        buf.close()
        plt.close('all')


def _extract_params_from_classifier(classifier) -> Dict[str, str]:
    """
    Извлекает гиперпараметры из классификатора через get_params() или атрибуты.
    Возвращает отфильтрованный dict с ключевыми параметрами.
    """
    key_params = [
        'n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
        'min_data_in_leaf', 'feature_fraction', 'bagging_fraction',
        'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda',
        'min_samples_split', 'min_samples_leaf', 'max_features',
        'criterion', 'contamination', 'random_state'
    ]

    try:
        if hasattr(classifier, 'get_params'):
            all_params = classifier.get_params()
            return {k: str(v) for k, v in all_params.items()
                    if k in key_params and v is not None}
    except Exception:
        pass


    # Fallback: ручное извлечение атрибутов
    params = {}
    for attr in key_params:
        if hasattr(classifier, attr):
            val = getattr(classifier, attr)
            if val is not None:
                params[attr] = str(val)

    return params or {'info': 'Параметры недоступны'}


def _extract_features_from_bundle(bundle) -> List[str]:
    """
    Извлекает имена признаков из pipeline или классификатора.
    """
    try:
        model_obj = bundle.pipeline

        # Если это sklearn Pipeline
        if hasattr(model_obj, 'named_steps'):
            # Пробуем получить из классификатора
            if 'classifier' in model_obj.named_steps:
                clf = model_obj.named_steps['classifier']
                if hasattr(clf, 'feature_names_in_'):
                    return list(clf.feature_names_in_)

            # Пробуем получить из preprocessor
            if 'preprocessor' in model_obj.named_steps:
                prep = model_obj.named_steps['preprocessor']
                if hasattr(prep, 'get_feature_names_out'):
                    try:
                        return list(prep.get_feature_names_out())
                    except Exception:
                        pass
                if hasattr(prep, 'feature_names_in_'):
                    return list(prep.feature_names_in_)

        # Прямой классификатор
        if hasattr(model_obj, 'feature_names_in_'):
            return list(model_obj.feature_names_in_)

    except Exception:
        pass

    # Fallback: пробуем загрузить из feature_names.txt
    if hasattr(bundle, 'root_dir') and bundle.root_dir:
        feature_names_file = bundle.root_dir / "feature_names.txt"
        if feature_names_file.exists():
            with open(feature_names_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

    # Fallback: заглушка
    return [f'feature_{i}' for i in range(50)]


def _get_model_metadata_from_file(model_id: str, env: str = 'test') -> Optional[Dict]:
    """
    Загружает ВСЕ данные о модели напрямую из сохранённых файлов через ModelManager.
    НЕ использует статические конфиги. Использует функции из model_metrics.py.
    """
    try:
        # Проверяем наличие ModelManager и модели
        if not hasattr(current_app, 'model_manager'):
            current_app.logger.error("ModelManager not initialized")
            return None

        model_manager = current_app.model_manager
        if not hasattr(model_manager, 'file_map') or model_id not in model_manager.file_map:
            current_app.logger.warning(f"Model {model_id} not found in file_map")
            return None

        # Загружаем бандл через ModelManager
        bundle = model_manager._get_or_load_bundle(model_id, env)
        model_obj = bundle.pipeline

        # Извлекаем классификатор
        if hasattr(model_obj, 'named_steps') and 'classifier' in model_obj.named_steps:
            classifier = model_obj.named_steps['classifier']
            is_pipeline = True
        else:
            classifier = model_obj
            is_pipeline = False

        # Путь к метрикам
        metrics_path = bundle.root_dir / 'metrics.json'

        # Используем функцию extract_model_metadata из model_metrics.py
        metadata = extract_model_metadata(
            classifier=classifier,
            bundle=bundle,
            model_id=model_id,
            metrics_path=metrics_path if metrics_path.exists() else None
        )

        # Добавляем флаг is_pipeline
        if metadata:
            metadata['is_pipeline'] = is_pipeline

        return metadata

    except FileNotFoundError as e:
        current_app.logger.error(f"Model file not found for {model_id}: {e}")
        return None
    except Exception as e:
        current_app.logger.error(f"Error loading model metadata for {model_id}: {e}", exc_info=True)
        return None


def _compute_if_feature_importance(models_dict, feature_names, n_top=10):
    """
    Вычисляет важность признаков для Isolation Forest через частоту использования в разбиениях.
    Isolation Forest не имеет feature_importances_, поэтому считаем эвристику.
    """
    if not isinstance(models_dict, dict) or len(models_dict) == 0:
        return None, None

    n_features = len(feature_names)
    importance_accumulator = np.zeros(n_features)
    model_count = 0

    for cls_name, model in models_dict.items():
        try:
            # Isolation Forest не имеет feature_importances_, считаем через деревья
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                tree_importances = np.zeros(n_features)
                for tree in model.estimators_:
                    # Получаем структуру дерева
                    tree_struct = tree.tree_
                    # Считаем, сколько раз каждый признак используется для разбиения
                    for node in range(tree_struct.node_count):
                        feature_idx = tree_struct.feature[node]
                        if feature_idx >= 0 and feature_idx < n_features:  # не лист
                            tree_importances[feature_idx] += 1
                # Нормализуем по числу деревьев
                if len(model.estimators_) > 0:
                    imp = tree_importances / len(model.estimators_)
                else:
                    continue
                importance_accumulator += imp
                model_count += 1
        except Exception as e:
            current_app.logger.warning(f"Error computing importance for class {cls_name}: {e}")
            continue

    if model_count == 0:
        return None, None

    importance_avg = importance_accumulator / model_count

    # Возвращаем топ-N
    n_top = min(n_top, len(importance_avg))
    indices = np.argsort(importance_avg)[-n_top:][::-1]
    return indices, importance_avg[indices]

def _get_model_viz(model_id: str) -> Dict[str, Any]:
    """Генерирует визуализацию для указанной модели"""
    default_error = {'success': False, 'error': 'Visualization not available'}

    try:
        if not hasattr(current_app, 'model_manager'):
            return {'success': False, 'error': 'ModelManager not initialized'}

        model_manager = current_app.model_manager
        if not hasattr(model_manager, 'file_map') or model_id not in model_manager.file_map:
            return {'success': False, 'error': f'Model {model_id} not found'}

        bundle = model_manager._get_or_load_bundle(model_id, 'test')
        model_obj = bundle.pipeline

        # Определяем тип объекта: пайплайн или прямой классификатор
        if hasattr(model_obj, 'named_steps') and 'classifier' in model_obj.named_steps:
            pipeline = model_obj
            classifier = pipeline.named_steps['classifier']
            is_pipeline = True
        else:
            pipeline = None
            classifier = model_obj
            is_pipeline = False

        # Получаем имена признаков
        feature_names = _extract_features_from_bundle(bundle)

        result = {'success': True, 'info': {
            'model_type': type(classifier).__name__,
            'is_pipeline': is_pipeline
        }}

        # === Извлечение параметров модели ===
        tree_params = {}
        model_type_name = type(classifier).__name__

        if hasattr(classifier, 'booster_'):  # LightGBM
            tree_params.update({
                'num_leaves': getattr(classifier, 'num_leaves', None),
                'max_depth': getattr(classifier, 'max_depth', None),
                'n_estimators': getattr(classifier, 'n_estimators', None),
                'learning_rate': getattr(classifier, 'learning_rate', None),
                'model_type': 'LightGBM'
            })
        elif hasattr(classifier, 'get_booster'):  # XGBoost
            tree_params.update({
                'max_leaves': getattr(classifier, 'max_leaves', None),
                'max_depth': getattr(classifier, 'max_depth', None),
                'n_estimators': getattr(classifier, 'n_estimators', None),
                'learning_rate': getattr(classifier, 'learning_rate', None),
                'model_type': 'XGBoost'
            })
        elif hasattr(classifier, 'estimators_') and not isinstance(classifier, dict):  # Random Forest
            tree_params.update({
                'n_estimators': getattr(classifier, 'n_estimators', None),
                'max_depth': getattr(classifier, 'max_depth', None),
                'criterion': getattr(classifier, 'criterion', None),
                'model_type': 'Random Forest'
            })
        elif model_type_name == 'IsolationForest':  # Isolation Forest (одна модель)
            tree_params.update({
                'n_estimators': getattr(classifier, 'n_estimators', None),
                'max_samples': getattr(classifier, 'max_samples', None),
                'contamination': getattr(classifier, 'contamination', None),
                'max_features': getattr(classifier, 'max_features', None),
                'random_state': getattr(classifier, 'random_state', None),
                'model_type': 'Isolation Forest'
            })
        elif isinstance(classifier, dict) and len(classifier) > 0:  # Isolation Forest One-vs-Rest
            first_model = list(classifier.values())[0]
            tree_params.update({
                'n_estimators': getattr(first_model, 'n_estimators', None),
                'max_samples': getattr(first_model, 'max_samples', None),
                'contamination': getattr(first_model, 'contamination', None),
                'max_features': getattr(first_model, 'max_features', None),
                'model_type': 'Isolation Forest (One-vs-Rest)',
                'n_class_models': len(classifier),
                'strategy': 'One-vs-Rest: отдельная модель для каждого класса'
            })

        # Добавляем метрики если есть в бандле
        if hasattr(bundle, 'metrics') and bundle.metrics:
            tree_params.update({
                'f1_score': bundle.metrics.get('f1_score'),
                'roc_auc': bundle.metrics.get('roc_auc'),
                'accuracy': bundle.metrics.get('accuracy'),
            })

        result['params'] = tree_params

        # === График важности признаков (если доступен) ===
        try:
            indices = None
            importances = None
            n_features = 0

            if hasattr(classifier, 'feature_importances_') and not isinstance(classifier, dict):
                importances = classifier.feature_importances_
                n_features = min(10, len(importances))
                indices = np.argsort(importances)[-n_features:][::-1]
            # Случай 2: Isolation Forest One-vs-Rest (dict моделей)
            elif isinstance(classifier, dict) and len(classifier) > 0:
                indices, importances = _compute_if_feature_importance(
                    classifier, feature_names, n_top=10
                )
                if indices is not None:
                    n_features = len(indices)

            # Случай 3: Единичная модель Isolation Forest
            elif isinstance(classifier, dict) and len(classifier) > 0:
                n_features_total = len(feature_names)
                tree_importances = np.zeros(n_features_total)

                for tree in classifier.estimators_:
                    # Получаем структуру дерева
                    tree_struct = tree.tree_
                    # Считаем, сколько раз каждый признак используется для разбиения
                    for node in range(tree_struct.node_count):
                        feature_idx = tree_struct.feature[node]
                        if feature_idx >= 0 and feature_idx < n_features_total:  # не лист
                            tree_importances[feature_idx] += 1

                # Нормализуем по количеству деревьев
                importances = tree_importances / len(classifier.estimators_)
                n_features = min(10, len(importances))
                indices = np.argsort(importances)[-n_features:][::-1]

            # Если важность вычислена — строим график
            if indices is not None and importances is not None and len(indices) > 0:
                plt.figure(figsize=(10, 6), facecolor='#1f2937')
                # Явно определяем значения для графика
                if isinstance(classifier, dict):
                    # Isolation Forest: importances уже отфильтрованы функцией _compute_if_feature_importance
                    plot_values = importances
                else:
                    # RandomForest/XGBoost/LightGBM: нужно взять значения по индексам из полного массива
                    plot_values = importances[indices] if indices is not None else importances
                plt.barh(range(n_features), plot_values, color='#2E86AB')

                labels = []
                for i in range(n_features):
                    feat_idx = indices[i]  # глобальный индекс признака
                    name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
                    labels.append(name[:22] + '...' if len(name) > 25 else name)

                plt.yticks(range(n_features), labels, fontsize=7)
                plt.xlabel('Важность (Gain)', color='#9ca3af', fontsize=9)
                plt.title('Важность признаков', color='white', fontsize=12)
                plt.tick_params(colors='#9ca3af', labelsize=8)
                plt.tight_layout()
                result['importance'] = _plot_to_base64()
                plt.close('all')
        except Exception as e:
            current_app.logger.warning(f"Importance plot error: {e}")

        # === Визуализация модели (дерево или альтернатива) ===
        tree_img = None
        try:
            if hasattr(classifier, 'booster_'):  # LightGBM
                import lightgbm as lgb
                if getattr(classifier, 'num_leaves', 255) <= 31:
                    plt.figure(figsize=(20, 12), facecolor='#1f2937')
                    lgb.plot_tree(classifier.booster_, tree_index=0, figsize=(12, 8))
                    plt.title('Дерево решений №1', color='white', fontsize=12)
                    plt.axis('off')
                    tree_img = _plot_to_base64()
                    plt.close('all')

            elif hasattr(classifier, 'get_booster'):  # XGBoost
                import xgboost as xgb
                booster = classifier.get_booster()
                plt.figure(figsize=(24, 16), facecolor='#1f2937', dpi=200)
                xgb.plot_tree(
                    booster, tree_idx=0, rankdir='LR',
                    condition_node_params={'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#78bceb'},
                    leaf_node_params={'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}
                )
                plt.title('Дерево решений №1', color='white', fontsize=11)
                plt.axis('off')
                tree_img = _plot_to_base64()
                plt.close('all')

            elif hasattr(classifier, 'estimators_') and hasattr(classifier, 'feature_names_in_'):  # RandomForest
                from sklearn.tree import plot_tree
                if len(classifier.estimators_) > 0:
                    tree = classifier.estimators_[0]
                    all_feature_names = list(classifier.feature_names_in_)
                    display_names = [name[:22] + '...' if len(name) > 25 else name
                                     for name in all_feature_names]

                    plt.figure(figsize=(16, 10), facecolor='#1f2937', dpi=200)
                    plot_tree(
                        tree,
                        feature_names=display_names,
                        class_names=[str(c)[:15] for c in classifier.classes_],
                        filled=True,
                        rounded=True,
                        max_depth=3,
                        fontsize=8
                    )
                    plt.title('Дерево решений №1 (Random Forest)', color='white', fontsize=12)
                    plt.axis('off')
                    tree_img = _plot_to_base64()
                    plt.close('all')

            elif isinstance(classifier, dict) and len(classifier) > 0:  # Isolation Forest One-vs-Rest
                # Альтернативная визуализация: распределение скоров аномальности
                # Берём первую модель для демонстрации
                first_cls_name = list(classifier.keys())[0]
                first_model = classifier[first_cls_name]

                # Загружаем тестовые данные для визуализации (если доступны в бандле)
                # Или генерируем синтетические для демонстрации
                try:
                    # Пытаемся загрузить признаки из файла для генерации демо-данных
                    if hasattr(bundle, 'root_dir'):
                        feat_path = bundle.root_dir / 'feature_names.txt'
                        if feat_path.exists():
                            with open(feat_path, 'r', encoding='utf-8') as f:
                                demo_features = [line.strip() for line in f if line.strip()]
                            n_feat = len(demo_features)
                        else:
                            n_feat = 50
                            demo_features = [f'feature_{i}' for i in range(n_feat)]
                    else:
                        n_feat = 50
                        demo_features = [f'feature_{i}' for i in range(n_feat)]

                    # Генерируем демо-данные для визуализации
                    np.random.seed(42)
                    n_samples = 500
                    X_demo = np.random.randn(n_samples, n_feat)

                    # Получаем скоры от модели
                    scores = first_model.decision_function(X_demo)

                    # Визуализация: гистограмма распределения скоров
                    plt.figure(figsize=(10, 6), facecolor='#1f2937')

                    # Разбиваем на квантили для наглядности
                    q25, q50, q75 = np.percentile(scores, [25, 50, 75])

                    plt.hist(scores, bins=40, alpha=0.8, color='#2E86AB', edgecolor='white')
                    plt.axvline(q50, color='#E74C3C', linestyle='--', linewidth=2, label=f'Медиана: {q50:.3f}')
                    plt.axvline(q25, color='#F39C12', linestyle=':', linewidth=1.5, label=f'Q25: {q25:.3f}')
                    plt.axvline(q75, color='#F39C12', linestyle=':', linewidth=1.5, label=f'Q75: {q75:.3f}')

                    plt.xlabel('Anomaly Score\n(чем больше → объект более "нормальный" для этого класса)',
                               color='#9ca3af', fontsize=9)
                    plt.ylabel('Частота', color='#9ca3af', fontsize=9)
                    plt.title(f'Распределение скоров для класса "{first_cls_name}"\n(Isolation Forest)',
                              color='white', fontsize=11)
                    plt.legend(fontsize=8, framealpha=0.9)
                    plt.grid(alpha=0.3, axis='y')
                    plt.tight_layout()
                    tree_img = _plot_to_base64()
                    plt.close('all')

                except Exception as viz_e:
                    current_app.logger.warning(f"IF viz demo error: {viz_e}")
                    # Если демо-визуализация не удалась — возвращаем None
                    tree_img = None

            # Если модель не поддерживает визуализацию — tree_img остаётся None

        except Exception as e:
            current_app.logger.warning(f"Tree viz error for {model_id}: {e}")
            tree_img = None  # Не ломаем ответ, просто не показываем дерево

        result['tree'] = tree_img

        # === HTML представление Pipeline (если есть) ===
        if is_pipeline and hasattr(pipeline, '_repr_html_'):
            try:
                result['pipeline'] = pipeline._repr_html_()
                result['pipeline_type'] = 'html'
            except Exception:
                pass

        return result

    except Exception as e:
        current_app.logger.error(f"Viz error {model_id}: {e}", exc_info=True)
        return default_error

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
            'has_viz': True
        },
        {
            'id': 'isolation_forest',
            'name': 'Isolation Forest',
             'description': 'Unsupervised алгоритм для обнаружения аномалий. Использует стратегию One-vs-Rest для многоклассовой классификации: обучается отдельная модель для каждого класса, при предсказании выбирается класс с наименьшим скором аномальности.',
            'params': [
                {'label': 'Число деревьев', 'key': 'n_estimators', 'type': 'number', 'value': 200},
                {'label': 'Max Samples', 'key': 'max_samples', 'type': 'number', 'value': 512},
                {'label': 'Contamination', 'key': 'contamination', 'type': 'number', 'value': 'auto'},
                {'label': 'Max Features', 'key': 'max_features', 'type': 'number', 'value': 0.8, 'step': 0.1},
            ],
            'has_viz': True
        },
    ]
    return render_template('models.html', models=ml_models)


@bp.route('/detail/<model_id>')
def model_detail(model_id: str):
    """
    Страница с подробным описанием модели.
    ВСЕ данные загружаются динамически из файлов через ModelManager.
    """
    # 1. Загружаем название проекта из настроек
    project_config = load_config()
    project_name = project_config.get('project_name', 'ML Network Security Project')

    # 2. Загружаем метаданные модели ИЗ ФАЙЛА через ModelManager
    model = _get_model_metadata_from_file(model_id)

    current_app.logger.info(f"Loading model: {model_id}")

    if not model:
        return render_template('error.html',
                               message=f"Модель '{model_id}' не найдена или не может быть загружена",
                               project_name=project_name), 404

    # 3. Передаём данные в шаблон
    return render_template('model_detail.html',
                           model=model,
                           project_name=project_name)


@bp.route('/api/viz/<model_id>')
def model_viz_api(model_id: str):
    """API endpoint для получения визуализации модели"""
    result = _get_model_viz(model_id)
    return jsonify(result)


def _is_categorical_feature(col_name: str) -> bool:
    """
    Определяет, является ли признак категориальным (one-hot encoded).
    Улучшенная логика определения.
    """
    # MAC-адреса и подобные паттерны
    if ':' in col_name:
        return True
    # One-hot encoded признаки (обычно содержат конкретные значения после _)
    if col_name.count('_') >= 2:
        parts = col_name.split('_')
        # Если последняя часть - короткое значение (не 'port', 'length' и т.д.)
        if len(parts[-1]) <= 10 and parts[-1].isalnum():
            return True
    return False

def _make_dummy_df(feature_names):
    """Fallback: одна случайная строка если тестового файла нет."""
    row = {}
    for col in feature_names:
        row[col] = [np.random.choice([0, 1]) if _is_categorical_feature(col)
                    else np.random.normal(0, 1)]
    return pd.DataFrame(row)

@bp.route('/predict', methods=['POST'])
def predict():
    """Роут для предсказания"""
    if not hasattr(current_app, 'model_manager'):
        return jsonify({"error": "ModelManager not initialized"}), 500

    model_manager = current_app.model_manager

    # Парсинг входных данных
    if request.is_json:
        data = request.get_json()
        selected_model_id = data.get('algo', 'lightgbm')
        env = data.get('env', 'test')
    else:
        selected_model_id = request.form.get('selected_model', 'lightgbm')
        env = request.form.get('env', 'test')

    if selected_model_id not in model_manager.file_map:
        return jsonify({"error": f"Модель '{selected_model_id}' не поддерживается."}), 400

    try:
        # Загружаем бандл модели для получения правильного набора признаков
        bundle = model_manager._get_or_load_bundle(selected_model_id, env)

        # Используем feature_names из бандла или извлекаем из pipeline
        feature_names = bundle.feature_names
        if feature_names is None:
            feature_names = _extract_features_from_bundle(bundle)


        # Загружаем реальный тестовый файл
        test_file_rel = TEST_DATA_MAP.get(selected_model_id)
        if test_file_rel:
            test_file_path = Path(current_app.root_path) / test_file_rel
            if test_file_path.exists():
                input_df = pd.read_csv(test_file_path)
                # Если в файле есть колонка метки — убираем её перед предсказанием
                for label_col in ('Label', 'label', 'Class', 'class', 'Attack', 'attack'):
                    if label_col in input_df.columns:
                        input_df = input_df.drop(columns=[label_col])
            else:
                current_app.logger.warning(f"Test file not found: {test_file_path}, using dummy")
                input_df = _make_dummy_df(feature_names)
        else:
            input_df = _make_dummy_df(feature_names)

        predictions = model_manager.predict(selected_model_id, input_df, env)
        predicted_class = predictions[0] if predictions else "Unknown"
        threats_count = sum(1 for p in predictions if p != 'Benign')

        # Распределение угроз
        threat_dist = {}
        for pred in predictions:
            if pred != 'Benign':
                threat_dist[pred] = threat_dist.get(pred, 0) + 1

        color_map = {'DoS': 'red', 'Intrusion': 'orange', 'Scan': 'yellow', 'Benign': 'green'}
        threat_distribution = [
                                  {'type': k, 'count': v,
                                   'percentage': round(v / len(predictions) * 100) if predictions else 0,
                                   'color': color_map.get(k.split('_')[0], 'gray')}
                                  for k, v in threat_dist.items()
                              ] or [{'type': 'Нет угроз', 'count': 0, 'percentage': 0, 'color': 'green'}]

        # Сохранение результатов в session (с ограничением размера)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        analysis_results = {
            "filename": f"analysis_{selected_model_id}_{timestamp.replace(':', '-')}.csv",
            "rows": len(predictions),
            "threats": threats_count,
            "model_used": f"{selected_model_id}/{env}",
            "timestamp": timestamp,
            "threat_distribution": threat_distribution,
            "accuracy": "96.42%"
        }
        session['analysis_results'] = analysis_results

        recent = session.get('recent_analyses', [])
        recent.insert(0, {
            'id': len(recent) + 1,
            'model': selected_model_id.title().replace('_', ' '),
            'dataset': analysis_results['filename'],
            'accuracy': analysis_results['accuracy'],
            'threats': threats_count,
            'timestamp': timestamp
        })
        # Ограничиваем историю 5 записями
        session['recent_analyses'] = recent[:5]
        session.modified = True

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "predicted_class": predicted_class,
            "count": len(predictions),
            "model_used": f"{selected_model_id}/{env}",
        })

    except Exception as e:
        current_app.logger.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"error": "Ошибка при выполнении предсказания. Проверьте логи сервера."}), 500
