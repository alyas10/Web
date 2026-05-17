# model_manager/model_metrics.py
"""
Модуль для расчета и извлечения характеристик моделей машинного обучения
из сохраненных pkl/joblib файлов (LightGBM, XGBoost, RandomForest и др.)
"""

import json
import os
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_model_params(classifier: Any) -> Dict[str, str]:
    """
    Извлекает гиперпараметры из классификатора через get_params() или атрибуты.
    Возвращает отфильтрованный dict с ключевыми параметрами.

    Args:
        classifier: Объект модели (LGBMClassifier, XGBClassifier, RandomForest и т.д.)

    Returns:
        Dict с параметрами модели
    """
    try:
        if hasattr(classifier, 'get_params'):
            all_params = classifier.get_params()
            # Расширенный список параметров для всех поддерживаемых моделей
            key_params = [
                # Общие параметры
                'n_estimators', 'max_depth', 'learning_rate', 'random_state',
                # Random Forest
                'min_samples_split', 'min_samples_leaf', 'max_features',
                'criterion', 'bootstrap', 'oob_score', 'class_weight', 'n_jobs',
                # LightGBM
                'num_leaves', 'min_data_in_leaf', 'feature_fraction', 'bagging_fraction',
                'bagging_freq', 'lambda_l1', 'lambda_l2', 'min_gain_to_split',
                'max_bin', 'feature_pre_filter', 'verbosity',
                # XGBoost
                'subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode',
                'gamma', 'reg_alpha', 'reg_lambda', 'scale_pos_weight',
                'min_child_weight', 'max_delta_step', 'tree_method', 'grow_policy',
                # Isolation Forest
                'contamination', 'max_samples', 'bootstrap', 'warm_start'
            ]
            result = {}
            for k, v in all_params.items():
                if k in key_params and v is not None:
                    # Форматируем значения для читаемого отображения
                    if isinstance(v, bool):
                        result[k] = 'Да' if v else 'Нет'
                    elif isinstance(v, (int, float)):
                        if isinstance(v, float) and v < 0.01 and v > 0:
                            result[k] = f'{v:.4f}'
                        else:
                            result[k] = str(v)
                    else:
                        result[k] = str(v)
            return result
    except Exception as e:
        pass

    params = {}
    for attr in ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                 'min_data_in_leaf', 'feature_fraction', 'subsample',
                 'colsample_bytree', 'criterion', 'contamination']:
        if hasattr(classifier, attr):
            val = getattr(classifier, attr)
            if val is not None:
                params[attr] = str(val)
    return params if params else {'info': 'Параметры недоступны'}


def extract_feature_names(bundle: Any) -> List[str]:
    """
    Извлекает имена признаков из pipeline или классификатора.

    Args:
        bundle: ArtifactBundle с pipeline

    Returns:
        Список имен признаков
    """
    try:
        model_obj = bundle.pipeline

        if hasattr(model_obj, 'named_steps'):
            if 'classifier' in model_obj.named_steps:
                clf = model_obj.named_steps['classifier']
                if hasattr(clf, 'feature_names_in_'):
                    return list(clf.feature_names_in_)

            if 'preprocessor' in model_obj.named_steps:
                prep = model_obj.named_steps['preprocessor']
                if hasattr(prep, 'get_feature_names_out'):
                    try:
                        return list(prep.get_feature_names_out())
                    except Exception:
                        pass
                if hasattr(prep, 'feature_names_in_'):
                    return list(prep.feature_names_in_)

        if hasattr(model_obj, 'feature_names_in_'):
            return list(model_obj.feature_names_in_)

    except Exception:
        pass

    return [f'feature_{i}' for i in range(50)]


def get_model_type_display(model_type: str) -> str:
    """Возвращает читаемое название типа модели для отображения."""
    type_mapping = {
        'LGBMClassifier': 'Градиентный бустинг (LightGBM)',
        'XGBClassifier': 'Градиентный бустинг (XGBoost)',
        'RandomForestClassifier': 'Ансамбль деревьев (Random Forest)',
        'IsolationForest': 'Детектор аномалий (Isolation Forest)',
        'GradientBoostingClassifier': 'Градиентный бустинг (sklearn)',
    }
    return type_mapping.get(model_type, model_type)


def get_model_speed(model_type: str) -> str:
    """Оценка скорости работы модели."""
    speed_map = {
        'LGBMClassifier': 'Высокая',
        'XGBClassifier': 'Средняя',
        'RandomForestClassifier': 'Средняя',
        'IsolationForest': 'Высокая',
        'GradientBoostingClassifier': 'Низкая',
    }
    return speed_map.get(model_type, 'N/A')


def get_model_complexity(model_type: str) -> str:
    """Оценка сложности модели."""
    complexity_map = {
        'LGBMClassifier': 'Средняя',
        'XGBClassifier': 'Высокая',
        'RandomForestClassifier': 'Низкая',
        'IsolationForest': 'Низкая',
        'GradientBoostingClassifier': 'Средняя',
    }
    return complexity_map.get(model_type, 'N/A')


def get_model_description(model_type: str) -> str:
    """Возвращает краткое описание модели."""
    descriptions = {
        'LGBMClassifier': 'Gradient boosting framework с высокой производительностью для больших датасетов.',
        'XGBClassifier': 'Оптимизированный gradient boosting алгоритм для сложных паттернов атак.',
        'RandomForestClassifier': 'Ансамбль деревьев решений. Устойчив к переобучению.',
        'IsolationForest': 'Unsupervised алгоритм для обнаружения аномалий и новых типов атак.',
        'GradientBoostingClassifier': 'Классический градиентный бустинг из sklearn.',
    }
    return descriptions.get(model_type, 'Модель машинного обучения для анализа сетевого трафика')


def get_model_advantages(model_type: str) -> List[str]:
    """Возвращает список преимуществ модели."""
    advantages = {
        'LGBMClassifier': [
            'Высокая скорость обучения и предсказания',
            'Эффективная работа с большими датасетами',
            'Низкое потребление памяти',
            'Поддержка категориальных признаков',
            'Встроенная регуляризация'
        ],
        'XGBClassifier': [
            'Высокая точность на сложных данных',
            'Встроенная обработка пропусков',
            'Гибкая настройка регуляризации',
            'Поддержка кастомных функций потерь'
        ],
        'RandomForestClassifier': [
            'Устойчивость к переобучению',
            'Работа с шумными данными',
            'Оценка важности признаков',
            'Минимальная настройка гиперпараметров'
        ],
        'IsolationForest': [
            'Не требует размеченных данных',
            'Обнаружение неизвестных угроз',
            'Низкая вычислительная сложность',
            'Масштабируемость на большие данные'
        ],
        'GradientBoostingClassifier': [
            'Хорошая интерпретируемость',
            'Гибкая настройка',
            'Работа с разнородными данными'
        ],
    }
    return advantages.get(model_type, ['Интерпретируемость', 'Автоматизация'])


def get_how_it_works(model_type: str) -> str:
    """Возвращает HTML описание того, как работает модель."""
    how_it_works = {
        'LGBMClassifier': '''
            <p><strong>LightGBM</strong> строит ансамбль деревьев последовательно,
            исправляя ошибки предыдущих итераций.</p>
            <h4 style="color: #2196F3; margin-top: 1rem;">Особенности:</h4>
            <ul>
                <li><strong>GOSS</strong> — выборка с большим градиентом для ускорения.</li>
                <li><strong>EFB</strong> — объединение взаимно исключающих признаков.</li>
                <li><strong>Leaf-wise рост</strong> — оптимальное разделение листьев.</li>
            </ul>
        ''',
        'XGBClassifier': '''
            <p><strong>XGBoost</strong> — оптимизированная реализация градиентного бустинга
            с регуляризацией и аппроксимацией второго порядка.</p>
        ''',
        'RandomForestClassifier': '''
            <p><strong>Random Forest</strong> строит множество деревьев на случайных
            подвыборках данных и признаков, усредняя их предсказания.</p>
        ''',
        'IsolationForest': '''
             <p><strong>Isolation Forest</strong> — алгоритм обнаружения аномалий, работающий по принципу
            изоляции объектов в пространстве признаков. Аномалии isolруются быстрее, чем нормальные объекты.</p>
            <h4 style="color: #2196F3; margin-top: 1rem;">Стратегия One-vs-Rest:</h4>
            <p>Для многоклассовой классификации используется стратегия <strong>One-vs-Rest</strong>:
            обучается отдельная модель Isolation Forest для каждого класса атак. При предсказании выбирается
            класс, для которого объект имеет <strong>наименьший скор аномальности</strong> (т.е. выглядит
            «наиболее нормальным» для данной модели).</p>
            <h4 style="color: #2196F3; margin-top: 1rem;">Особенности:</h4>
            <ul>
                <li><strong>Z-score нормализация</strong> скоров для сопоставимости между моделями.</li>
                <li><strong>Не требует размеченных данных</strong> для обучения каждой модели.</li>
                <li><strong>Эффективное обнаружение</strong> новых типов атак.</li>
            </ul>
        ''',
        'GradientBoostingClassifier': '''
            <p><strong>Gradient Boosting</strong> последовательно строит деревья,
            где каждое следующее исправляет ошибки предыдущих.</p>
        ''',
    }
    return how_it_works.get(model_type, '<p>Модель обучена на данных сетевого трафика.</p>')


def get_security_application(model_type: str) -> str:
    """Возвращает HTML описание применения модели в безопасности."""
    security_app = {
        'LGBMClassifier': '''
            <p>Применяется для обнаружения DDoS-атак, классификации вторжений
            и детектирования аномалий в реальном времени.</p>
        ''',
        'XGBClassifier': '''
            <p>Эффективен для мультиклассовой классификации атак и обработки
            несбалансированных данных через scale_pos_weight.</p>
        ''',
        'RandomForestClassifier': '''
            <p>Подходит для базовой классификации атак с хорошей интерпретируемостью
            через feature importance.</p>
        ''',
        'IsolationForest': '''
            <p>Идеален для обнаружения <strong>новых, неизвестных типов атак</strong>
            без необходимости предварительной разметки данных. Стратегия One-vs-Rest позволяет
            эффективно классифицировать множественные типы атак, обучая отдельную модель
            для каждого класса.</p>
        ''',
        'GradientBoostingClassifier': '''
            <p>Применяется для классификации сетевых атак и анализа трафика.</p>
        ''',
    }
    return security_app.get(model_type, '<p>Применяется для анализа сетевой безопасности.</p>')


def has_tree_viz(model_type: str) -> bool:
    """Проверяет, поддерживает ли модель визуализацию дерева решений."""
    return model_type in ['LGBMClassifier', 'XGBClassifier','RandomForestClassifier']


def _extract_accuracy_from_model_attributes(classifier: Any) -> Optional[float]:
    """
    Пытается извлечь точность из атрибутов самой модели.
    Возвращает float в диапазоне [0, 1] или None если не удалось.
    """
    # LightGBM/XGBoost: best_score_
    if hasattr(classifier, 'best_score_') and classifier.best_score_ is not None:
        score = classifier.best_score_
        if isinstance(score, dict):
            for dataset_metrics in score.values():
                if isinstance(dataset_metrics, dict):
                    for metric_name, value in dataset_metrics.items():
                        if 'error' in metric_name.lower():
                            return 1.0 - float(value)
                        elif 'acc' in metric_name.lower() or 'auc' in metric_name.lower():
                            return float(value)
        else:
            return float(score)

    # RandomForest: oob_score_
    if hasattr(classifier, 'oob_score_') and classifier.oob_score_ is not None:
        return float(classifier.oob_score_)

    # IsolationForest: эвристика на основе contamination
    if hasattr(classifier, 'contamination'):
        return 1.0 - float(classifier.contamination)

    return None


def _calculate_accuracy_on_sample(classifier: Any, X_sample: pd.DataFrame,
                                  y_sample: pd.Series) -> Optional[float]:
    """
    Рассчитывает точность на предоставленной выборке.
    Возвращает float в диапазоне [0, 1] или None если не удалось.
    """
    if not SKLEARN_AVAILABLE:
        return None

    try:
        y_pred = classifier.predict(X_sample)

        if len(np.unique(y_sample)) > 2:
            return accuracy_score(y_sample, y_pred)
        else:
            return accuracy_score(y_sample, y_pred)
    except Exception:
        return None


def _save_metrics_to_file(metrics: Dict[str, float], metrics_path: Any) -> bool:
    """
    Сохраняет метрики в JSON файл.
    Возвращает True если успешно, False иначе.
    """
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def _load_metrics_from_file(metrics_path: Any) -> Optional[Dict[str, float]]:
    """
    Загружает метрики из JSON файла.
    Возвращает dict с метриками или None если не удалось.
    """
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def calculate_and_save_metrics(
        classifier: Any,
        bundle: Any,
        metrics_path: Any,
        X_sample: Optional[pd.DataFrame] = None,
        y_sample: Optional[pd.Series] = None
) -> Optional[Dict[str, float]]:
    """
    Рассчитывает метрики модели и сохраняет их в metrics.json.

    Args:
        classifier: Обученная модель
        bundle: ArtifactBundle с пайплайном
        metrics_path: Путь к файлу metrics.json (pathlib.Path)
        X_sample: Опциональная выборка для оценки
        y_sample: Опциональные истинные метки для выборки

    Returns:
        Dict с рассчитанными метриками или None если расчет не удался
    """
    metrics = {}

    # 1. Пробуем взять встроенные метрики модели
    accuracy = _extract_accuracy_from_model_attributes(classifier)
    if accuracy is not None:
        metrics['accuracy'] = round(accuracy, 4)

    # 2. Если встроенных метрик нет — считаем на выборке
    elif X_sample is not None and y_sample is not None and len(X_sample) > 0:
        accuracy = _calculate_accuracy_on_sample(classifier, X_sample, y_sample)
        if accuracy is not None:
            metrics['accuracy'] = round(accuracy, 4)

    # Если не удалось рассчитать ни одним способом — возвращаем None
    if not metrics:
        return None

    # Сохраняем в файл
    _save_metrics_to_file(metrics, metrics_path)

    return metrics


def calculate_accuracy_from_metrics(metrics_path: Any) -> Optional[str]:
    """
    Читает точность модели из файла metrics.json.
    Возвращает строку формата '~XX.X%' или None если не удалось.
    """
    if not hasattr(metrics_path, 'exists') or not metrics_path.exists():
        return None

    metrics = _load_metrics_from_file(metrics_path)
    if metrics is None:
        return None

    acc = metrics.get('accuracy')
    if acc is None:
        return None

    try:
        acc_float = float(acc)
        if acc_float > 1:
            return f"~{round(acc_float, 1)}%"
        else:
            return f"~{round(acc_float * 100, 1)}%"
    except (ValueError, TypeError):
        return None


def extract_model_metadata(
        classifier: Any,
        bundle: Any,
        model_id: str,
        metrics_path: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Извлекает все метаданные модели из классификатора и bundle.

    Args:
        classifier: Объект модели
        bundle: ArtifactBundle с pipeline и информацией
        model_id: Идентификатор модели
        metrics_path: Опциональный путь к файлу метрик

    Returns:
        Dict со всеми метаданными для отображения в model_detail.html
    """
    model_type = type(classifier).__name__

    # Расчет точности
    accuracy = None

    if metrics_path and hasattr(metrics_path, 'exists'):
        # Сначала пробуем загрузить из существующего файла
        if metrics_path.exists():
            accuracy = calculate_accuracy_from_metrics(metrics_path)

        # Если файла нет или не удалось загрузить — рассчитываем и сохраняем
        if accuracy is None:
            # Пробуем получить тестовые данные из bundle (если есть)
            X_sample, y_sample = None, None
            if hasattr(bundle, 'X_test') and hasattr(bundle, 'y_test'):
                X_sample = bundle.X_test[:100]
                y_sample = bundle.y_test[:100]

            calculated_metrics = calculate_and_save_metrics(
                classifier=classifier,
                bundle=bundle,
                metrics_path=metrics_path,
                X_sample=X_sample,
                y_sample=y_sample
            )

            if calculated_metrics and 'accuracy' in calculated_metrics:
                acc_value = calculated_metrics['accuracy']
                accuracy = f"~{round(acc_value * 100, 1)}%" if acc_value <= 1 else f"~{round(acc_value, 1)}%"

    # Форматируем точность для отображения
    accuracy_display = accuracy if accuracy is not None else 'N/A'

    return {
        'id': model_id,
        'name': model_type.replace('Classifier', '').replace('Forest', ' Forest'),
        'tagline': get_model_description(model_type),
        'type': get_model_type_display(model_type),
        'speed': get_model_speed(model_type),
        'accuracy': accuracy_display,
        'complexity': get_model_complexity(model_type),
        'params': extract_model_params(classifier),
        'features': extract_feature_names(bundle)[:10],
        'has_tree_viz': has_tree_viz(model_type),
        'how_it_works': get_how_it_works(model_type),
        'security_application': get_security_application(model_type),
        'advantages': get_model_advantages(model_type),
        'source': 'loaded_from_file',
        'model_type_raw': model_type,
        'is_pipeline': hasattr(bundle.pipeline, 'named_steps'),
    }
