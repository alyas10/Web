"""
test/test_demo.py
Запуск: python -m pytest test/test_demo.py -v
"""
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ВАЖНО: Импортируем NumericFeatureSelector до загрузки пайплайна
from model_manager.model_utils import NumericFeatureSelector


def test_data_loader_csv():
    """Тест 1: Загрузка CSV данных через DataPipelineAdapter"""
    from data_loader.base import DataPipelineAdapter

    # Создаем тестовые данные
    test_data = {
        'Time': [1.0, 2.0, 3.0],
        'Length': [100, 200, 150],
        'IP TTL': [64, 128, 32],
        'TCP Source Port': [80, 443, 22],
        'Protocol_TCP': [1, 1, 0],
        'Source': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'Destination': ['8.8.8.8', '1.1.1.1', '9.9.9.9']
    }
    df = pd.DataFrame(test_data)

    # Инициализируем адаптер
    adapter = DataPipelineAdapter(
        expected_features=list(test_data.keys()),
        numeric_features=['Time', 'Length', 'IP TTL', 'TCP Source Port', 'Protocol_TCP'],
        categorical_features=['Source', 'Destination']
    )

    # Выполняем подготовку данных
    prepared_df = adapter.prepare(df)

    assert not prepared_df.empty, "DataFrame не должен быть пустым"
    assert len(prepared_df) == 3, "Должно остаться 3 строки"


def test_feature_selector():
    """Тест 2: Отбор числовых признаков NumericFeatureSelector"""
    from model_manager.model_utils import NumericFeatureSelector

    # Создаем тестовые данные
    np.random.seed(42)
    X = pd.DataFrame({
        'feat1': np.random.rand(100),
        'feat2': np.random.rand(100),
        'feat3': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    y = X['target']
    X = X.drop('target', axis=1)

    # Инициализируем селектор
    selector = NumericFeatureSelector(k=2)
    selector.fit(X, y)
    X_selected = selector.transform(X)

    assert X_selected.shape[1] == 2, f"Должно быть отобрано 2 признака, получено {X_selected.shape[1]}"


def test_xgboost_model_loading():
    """Тест 3: Загрузка модели через ModelManager и проверка метрик"""
    from model_manager.model_manager import ModelManager
    import json

    model_path = "pipeline/xgboost/test/xgb_optuna_best.joblib"
    metrics_path = "pipeline/xgboost/test/metrics.json"
    meta_path = "pipeline/xgboost/test/meta.json"

    # Проверяем существование файлов
    assert os.path.exists(model_path), f"Файл модели не найден: {model_path}"
    assert os.path.exists(metrics_path), f"Файл метрик не найден: {metrics_path}"
    assert os.path.exists(meta_path), f"Файл meta.json не найден: {meta_path}"

    # Загружаем метрики
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    assert 'accuracy' in metrics, "Отсутствует accuracy в метриках"
    assert 'roc_auc_macro' in metrics, "Отсутствует roc_auc_macro в метриках"
    assert metrics['accuracy'] > 0.95, f"Accuracy слишком низкий: {metrics['accuracy']}"
    assert metrics['roc_auc_macro'] > 0.99, f"ROC-AUC слишком низкий: {metrics['roc_auc_macro']}"

    # Загрузка через ModelManager
    manager = ModelManager(models_root=PROJECT_ROOT / "pipeline")
    bundle = manager._get_or_load_bundle("xgboost", "test")

    assert bundle.pipeline is not None, "Pipeline не загружен"
    assert bundle.label_encoder is not None, "LabelEncoder не загружен"
    assert bundle.feature_names is not None, "Имена признаков не загружены"


def test_xgboost_prediction_with_manager():
    """Тест 4: Проверка предсказания через ModelManager с авто-нормализацией"""
    from model_manager.model_manager import ModelManager

    # Создаем ModelManager
    manager = ModelManager(models_root=PROJECT_ROOT / "pipeline")

    # Загружаем бандл для получения feature_names
    bundle = manager._get_or_load_bundle("xgboost", "test")
    feature_names = bundle.feature_names

    assert feature_names is not None, "Feature names не загружены"
    assert len(feature_names) > 0, "Список признаков пуст"

    # Создаем dummy-данные (могут быть в любом порядке, с лишними/отсутствующими колонками)
    np.random.seed(42)
    dummy_row = {}

    # Добавляем некоторые признаки в случайном порядке
    for col in feature_names[:10]:  # Берём первые 10 признаков для теста
        if col in ['TCP Destination Port', 'UDP Source Port', 'TCP Source Port']:
            dummy_row[col] = np.random.randint(1, 65535)
        elif col in ['IP TTL']:
            dummy_row[col] = np.random.randint(32, 128)
        elif col in ['Length', 'IP Length', 'TCP Length']:
            dummy_row[col] = np.random.randint(64, 1500)
        elif 'Flag' in col or 'Protocol' in col or 'Type' in col:
            dummy_row[col] = np.random.choice([0, 1])
        else:
            dummy_row[col] = np.random.rand() * 100

    # Добавляем лишнюю колонку (должна быть удалена при нормализации)
    dummy_row['extra_column'] = 'should_be_removed'

    dummy_df = pd.DataFrame([dummy_row])

    # Предсказание через ModelManager
    predictions = manager.predict(algo="xgboost", data=dummy_df, env="test")

    # Проверяем результат
    assert isinstance(predictions, list), "Предсказания должны быть списком"
    assert len(predictions) == 1, "Должно быть одно предсказание"
    assert isinstance(predictions[0], str), "Предсказание должно быть строкой"

    # Проверяем, что предсказание - валидный класс
    assert predictions[0] in bundle.label_encoder.classes_, \
        f"Предсказание '{predictions[0]}' не в списке классов"


def test_pcap_loader_structure():
    """Тест 5: Проверка структуры PCAP загрузчика"""
    from data_loader.pcap_loader import PcapScapyDataLoader

    # Создаем экземпляр загрузчика
    loader = PcapScapyDataLoader(max_packets=100, extract_features=True)

    # Проверяем атрибуты
    assert hasattr(loader, 'supported_extensions'), "Отсутствует supported_extensions"
    assert '.pcap' in loader.supported_extensions, "PCAP не поддерживается"
    assert hasattr(loader, 'load'), "Отсутствует метод load"
    assert hasattr(loader, '_extract_ml_features'), "Отсутствует метод извлечения признаков"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])