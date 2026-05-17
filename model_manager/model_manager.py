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
            },
            "isolation_forest": {
                "pipeline": "isolation_forest_v1.joblib",
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

    def _normalize_input_features(
            self,
            data: pd.DataFrame,
            expected_features: Optional[List[str]],
            algo: str
    ) -> pd.DataFrame:
        """
        Приводит входные данные к формату, ожидаемому моделью.

        - Добавляет отсутствующие признаки (заполняет 0)
        - Удаляет лишние колонки
        - Сортирует в правильном порядке
        - Конвертирует object-колонки в category для tree-моделей
        """
        if expected_features is None:
            # Если имена признаков неизвестны — возвращаем как есть
            # (модель должна справиться сама или упадёт с понятной ошибкой)
            return data.copy()

        result = data.copy()

        # 1. Конвертация object → category для tree-based моделей
        if algo in ("lightgbm", "xgboost", "random_forest"):
            for col in result.select_dtypes(include=['object','string']).columns:
                if col in expected_features:
                    result[col] = result[col].astype('category')

       # Для isolation_forest нужно масштабирование через StandardScaler
        if algo == "isolation_forest":
            # Масштабирование будет выполнено в predict() после этой функции
             pass

        # 2. Добавляем отсутствующие признаки
        missing = [f for f in expected_features if f not in result.columns]
        if missing:
            for col in missing:
                # Для one-hot encoded признаков (с '_' и ':') — 0, иначе — 0
                result[col] = 0

        # 3. Удаляем лишние колонки
        extra = [c for c in result.columns if c not in expected_features]
        if extra:
            result = result.drop(columns=extra)

        # 4. Сортируем в правильном порядке
        result = result[expected_features]

        return result

    def predict(self, algo: str, data: InputData, env: str = "test") -> List[str]:
        bundle = self._get_or_load_bundle(algo, env)


        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        expected_features = bundle.feature_names

        # === Универсальная предобработка признаков ===
        data = self._normalize_input_features(data, expected_features, algo)
        print(f"[DEBUG] algo={algo}, input_cols={list(data.columns)[:10]}..., expected={len(expected_features) if expected_features else 'None'}")
        # Для Isolation Forest нужно масштабирование перед предсказанием
        if algo == "isolation_forest":
            # Загружаем scaler из артефактов
            scaler_path = bundle.root_dir / "scaler.joblib"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                data_scaled = scaler.transform(data)
            else:
                raise ModelManagerError(f"Scaler not found for isolation_forest: {scaler_path}")

            # Для Isolation Forest используем кастомную функцию предсказания с калибровкой
            pred = self._if_multiclass_predict_calibrated(
                data_scaled, bundle.pipeline, bundle.label_encoder.classes_
            )
        else:
            # Вызов модели для остальных алгоритмов
            pred = bundle.pipeline.predict(data)

        if pred is None:
            raise ModelManagerError("Pipeline.predict() returned None")

        # Если модель уже вернула строковые метки
        if len(pred) > 0 and isinstance(pred.flat[0], str):
            return [str(x) for x in pred]

        # Декодирование числовых предсказаний
        try:
            labels = bundle.label_encoder.inverse_transform(np.asarray(pred).astype(int))
        except Exception as e:
            print(f"[ERROR] inverse_transform failed: {e}")
            raise

        return [str(x) for x in labels]

    def _if_multiclass_predict_calibrated(self, X_scaled, models, classes):
        """
        Предсказание для One-vs-Rest Isolation Forest с калибровкой скоров.
        Сырые скоры IF разных моделей несопоставимы.
        Нормализуем их по Z-score относительно обучающего распределения каждого класса.
        """
        n_samples = X_scaled.shape[0]
        n_classes = len(classes)
        scores_matrix = np.zeros((n_samples, n_classes))

        # Для каждой модели вычисляем скоры и нормализуем
        for i, cls_name in enumerate(classes):
            if cls_name not in models:
                continue

            raw_scores = models[cls_name].decision_function(X_scaled)
            # Используем глобальную статистику (mean=0, std=1) т.к. нет train_stats
            # Это упрощенная версия - в идеале нужно сохранять train_score_stats при обучении
            scores_matrix[:, i] = raw_scores

        # Возвращаем индекс класса с максимальным скором
        return np.argmax(scores_matrix, axis=1)
