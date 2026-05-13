# model_manager/model_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, List, Optional, Union

import joblib
import numpy as np


try:
    import pandas as pd
except Exception:
    raise RuntimeError("pandas required")

InputData = Union["pd.DataFrame", "np.ndarray"]


class ModelManagerError(Exception):
    pass


@dataclass(frozen=True)
class ArtifactBundle:
    pipeline: Any  # Pipeline
    label_encoder: Any
    root_dir: Path
    feature_names: Optional[List[str]] = None


class ModelManager:
    def __init__(self, models_root: Optional[Union[str, Path]] = None):
        self._lock = RLock()
        self._cache = {}
        self.models_root = Path(models_root or Path(__file__).parent.parent / "pipeline")

        self.file_map = {
            "lightgbm": {
                "pipeline": "full_pipeline.pkl",
                "label_encoder": "label_encoder_full_dataset.pkl",
            },
            "xgboost": {
                "pipeline": "xgb_optuna_best.joblib",
                "label_encoder": "label_encoder.joblib",
            },
            "random_forest": {
                "pipeline": "random_forest_v1.joblib",
                "label_encoder": "label_encoder.joblib",
            }
        }

    def _get_or_load_bundle(self, algo: str, env: str) -> ArtifactBundle:
        key = (algo, env)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            bundle = self._load_bundle(algo, env)
            self._cache[key] = bundle
            return bundle

    def _load_bundle(self, algo: str, env: str) -> ArtifactBundle:
        mapping = self.file_map[algo]
        root_dir = self.models_root / algo / env
        paths = {k: root_dir / v for k, v in mapping.items()}

        for p in paths.values():
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")

        pipeline = joblib.load(paths["pipeline"])
        label_encoder = joblib.load(paths["label_encoder"])

        # Загружаем имена признаков из доступных источников
        feature_names = self._load_feature_names(algo, root_dir, pipeline)

        return ArtifactBundle(
            pipeline=pipeline,
            label_encoder=label_encoder,
            root_dir=root_dir,
            feature_names=feature_names,
        )

    def _load_feature_names(self, algo: str, root_dir: Path, pipeline: Any) -> Optional[List[str]]:
        """Загружает имена признаков для модели из различных источников."""
        # 1. Пробуем загрузить из текстового файла feature_names.txt
        feature_names_file = root_dir / "feature_names.txt"
        if feature_names_file.exists():
            with open(feature_names_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        # 2. Пробуем загрузить из feature_info.pkl (для LightGBM)
        feature_info_file = root_dir / "feature_info.pkl"
        if feature_info_file.exists():
            try:
                import pickle
                with open(feature_info_file, 'rb') as f:
                    feature_info = pickle.load(f)
                if isinstance(feature_info, dict):
                    numeric = feature_info.get('numeric_features', [])
                    categorical = feature_info.get('categorical_features', [])
                    return numeric + categorical
            except Exception:
                pass

        # 3. Для XGBoost пробуем извлечь из meta.json (если есть class_names и features_count)
        meta_file = root_dir / "meta.json"
        if meta_file.exists() and algo == "xgboost":
            try:
                import json
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # Если есть информация о количестве признаков, но нет имен - возвращаем None
                # Имена будут извлечены из pipeline ниже
            except Exception:
                pass

        # 4. Пробуем извлечь из самого pipeline
        try:
            if hasattr(pipeline, 'named_steps'):
                # Если это sklearn Pipeline
                if 'classifier' in pipeline.named_steps:
                    clf = pipeline.named_steps['classifier']
                    if hasattr(clf, 'feature_names_in_'):
                        return list(clf.feature_names_in_)
                if 'preprocessor' in pipeline.named_steps:
                    prep = pipeline.named_steps['preprocessor']
                    if hasattr(prep, 'get_feature_names_out'):
                        try:
                            return list(prep.get_feature_names_out())
                        except Exception:
                            pass
                    if hasattr(prep, 'feature_names_in_'):
                        return list(prep.feature_names_in_)

            # Прямой классификатор (XGBoost, LightGBM)
            if hasattr(pipeline, 'feature_names_in_'):
                return list(pipeline.feature_names_in_)
        except Exception:
            pass

        # Fallback: возвращаем None, будем использовать заглушку
        return None

    def predict(self, algo: str, data: InputData, env: str = "test") -> List[str]:
        bundle = self._get_or_load_bundle(algo, env)

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        # Получаем ожидаемые имена признаков из бандла
        expected_features = bundle.feature_names

        # === Обработка данных для разных алгоритмов ===
        if algo == "xgboost":
            # XGBoost требует enable_categorical=True для категориальных данных
            # Конвертируем строковые колонки в category тип
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].astype('category')

            # Если известны ожидаемые признаки - приводим данные к ним
            if expected_features is not None:
                # Добавляем отсутствующие колонки с нулями
                missing_cols = [col for col in expected_features if col not in data.columns]
                if missing_cols:
                    for col in missing_cols:
                        # Определяем тип колонки по имени (one-hot encoded или numeric)
                        if '_' in col and ':' in col:
                            # Это one-hot encoded признак (например, MAC-адрес)
                            data[col] = 0
                        else:
                            data[col] = 0

                # Удаляем лишние колонки и сортируем в правильном порядке
                extra_cols = [col for col in data.columns if col not in expected_features]
                if extra_cols:
                    data = data.drop(columns=extra_cols)
                data = data[expected_features]

        elif algo == "random_forest":
            # RandomForest требует строгого соответствия признаков
            if expected_features is not None:
                processed_data = {}
                for col in expected_features:
                    if col in data.columns:
                        processed_data[col] = data[col].values
                    else:
                        # Если колонка отсутствует - заполняем нулями
                        processed_data[col] = np.zeros(len(data))
                data = pd.DataFrame(processed_data, index=data.index)

        elif algo == "lightgbm":
            # LightGBM может работать с категориальными данными
            # Конвертируем строковые колонки в category тип
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].astype('category')

            # Если известны ожидаемые признаки - приводим данные к ним
            if expected_features is not None:
                # Добавляем отсутствующие колонки
                missing_cols = [col for col in expected_features if col not in data.columns]
                if missing_cols:
                    for col in missing_cols:
                        data[col] = 0

                # Удаляем лишние колонки
                extra_cols = [col for col in data.columns if col not in expected_features]
                if extra_cols:
                    data = data.drop(columns=extra_cols)

                # Сортируем в правильном порядке
                data = data[expected_features]

        # Вызов модели вынесен из try, чтобы отделить ошибки инференса от ошибок декодирования
        pred = bundle.pipeline.predict(data)
        print(
            f"[DEBUG] pred type: {type(pred)}, shape: {getattr(pred, 'shape', 'N/A')}, value: {pred[:5] if hasattr(pred, '__len__') else 'N/A'}"
        )

        if pred is None:
            raise ModelManagerError("Pipeline.predict() returned None")

        # Если модель уже вернула строковые метки, обратное преобразование не требуется
        if len(pred) > 0 and isinstance(pred.flat[0], str):
            return [str(x) for x in pred]

        # Если вернула числовые коды, безопасно преобразуем и декодируем
        try:
            labels = bundle.label_encoder.inverse_transform(np.asarray(pred).astype(int))
        except Exception as e:
            print(f"[ERROR] inverse_transform failed: {e}")
            raise

        return [str(x) for x in labels]
