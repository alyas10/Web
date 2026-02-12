# model_manager/model_manager.py
"""
ModelManager: загрузка ML-моделей и артефактов препроцессинга с кэшированием.

Модуль рассчитан на использование в Flask-приложении для анализа сетевого трафика:
- модели и артефакты лежат на диске в структуре: models/{algo}/{env}/
- поддерживается несколько алгоритмов через словарь сопоставления имён файлов
- загрузка кэшируется, чтобы не читать .pkl при каждом запросе
- единая точка входа: ModelManager.predict(algo, data, env='test') -> list[str]

Типовой пайплайн:
1) Импутация (numeric_imputer + categorical_imputer, если есть)
2) Feature selection (feature_selector, если есть)
3) model.predict(...)
4) Для multiclass вероятностей -> argmax
5) label_encoder.inverse_transform(...) -> list[str]

Примечания по устойчивости:
- Артефакты могут быть sklearn-объектами (SimpleImputer, Selector, LabelEncoder),
  либо любыми объектами с методом transform / inverse_transform.
- Для imputers предпочтительно использовать сохранённые feature_names_in_ (если есть),
  иначе столбцы определяются эвристикой через pandas dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

import joblib
import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Для ModelManager требуется pandas (используется для работы с табличными признаками)."
    ) from exc


JSONLike = Union[Dict[str, Any], List[Dict[str, Any]]]
InputData = Union["pd.DataFrame", "np.ndarray", JSONLike]


class ModelManagerError(Exception):
    """Базовое исключение ModelManager."""


class UnsupportedAlgorithmError(ModelManagerError):
    """Алгоритм не поддержан (нет маппинга файлов)."""


class ArtifactNotFoundError(ModelManagerError):
    """Не найден один или несколько файлов артефактов."""


@dataclass(frozen=True)
class ArtifactBundle:
    """Набор загруженных объектов для конкретной пары (algo, env)."""

    model: Any
    label_encoder: Any
    feature_selector: Any
    numeric_imputer: Any
    categorical_imputer: Any
    root_dir: Path


class ModelManager:
    """
    Менеджер моделей/артефактов с кэшем.

    Parameters
    ----------
    models_root:
        Путь до директории `models/`. По умолчанию пытается найти `models/`
        рядом с корнем проекта (относительно текущего файла).
    file_map:
        Сопоставление {algo: {artifact_key: file_name}}.
        artifact_key должен быть одним из:
        - model
        - label_encoder
        - feature_selector
        - numeric_imputer
        - categorical_imputer
    """

    # Ключи артефактов, которые ожидаются менеджером
    _ARTIFACT_KEYS: Tuple[str, ...] = (
        "model",
        "label_encoder",
        "feature_selector",
        "numeric_imputer",
        "categorical_imputer",
    )

    def __init__(
        self,
        models_root: Optional[Union[str, Path]] = None,
        file_map: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> None:
        self._lock = RLock()
        self._cache: MutableMapping[Tuple[str, str], ArtifactBundle] = {}

        if models_root is None:
            # model_manager/model_manager.py -> model_manager/ -> project_root/
            project_root = Path(__file__).resolve().parents[1]
            models_root = project_root / "models"

        self.models_root = Path(models_root).resolve()

        # Маппинг по умолчанию — ваш точный кейс для LightGBM
        default_map: Dict[str, Dict[str, str]] = {
            "lightgbm": {
                "model": "lightgbm_full_dataset_model.pkl",
                "label_encoder": "label_encoder_full_dataset.pkl",
                "feature_selector": "feature_selector_full_dataset.pkl",
                "numeric_imputer": "numeric_imputer_full_dataset.pkl",
                "categorical_imputer": "categorical_imputer_full_dataset.pkl",
            },
            # Примеры заготовок: добавьте свои точные имена файлов под каждый алгоритм
            # "xgboost": {...},
            # "random_forest": {...},
        }

        if file_map is not None:
            # Объединяем: пользовательские значения перекрывают дефолтные
            merged: Dict[str, Dict[str, str]] = {**default_map}
            for algo, mapping in file_map.items():
                merged[algo] = {**merged.get(algo, {}), **dict(mapping)}
            self.file_map = merged
        else:
            self.file_map = default_map

    # ---------------------------
    # Public API
    # ---------------------------

    def predict(self, algo: str, data: InputData, env: str = "test") -> List[str]:
        """
        Выполнить предсказание для указанного алгоритма и окружения.

        Parameters
        ----------
        algo:
            Название алгоритма (например, "lightgbm").
        data:
            Входные признаки:
            - pandas.DataFrame (предпочтительно),
            - numpy.ndarray,
            - list[dict] (например, JSON из Flask request.get_json()),
            - dict[str, Any] (одна запись или column-oriented).
        env:
            Окружение: "test" / "production" (или другое, если есть директория).

        Returns
        -------
        list[str]
            Список строковых меток классов.
        """
        bundle = self._get_or_load_bundle(algo=algo, env=env)

        X = self._to_dataframe(data)
        X = self._apply_imputation(X, bundle.numeric_imputer, bundle.categorical_imputer)
        X_transformed = self._apply_feature_selection(X, bundle.feature_selector)

        raw_pred = self._model_predict(bundle.model, X_transformed)
        class_indices = self._normalize_predictions_to_class_indices(
            raw_pred=raw_pred,
            label_encoder=bundle.label_encoder,
        )

        labels = self._decode_labels(bundle.label_encoder, class_indices)
        return [str(x) for x in labels]

    def clear_cache(self) -> None:
        """Очистить кэш загруженных моделей/артефактов."""
        with self._lock:
            self._cache.clear()

    # ---------------------------
    # Loading / caching
    # ---------------------------

    def _get_or_load_bundle(self, algo: str, env: str) -> ArtifactBundle:
        key = (algo, env)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            bundle = self._load_bundle(algo=algo, env=env)
            self._cache[key] = bundle
            return bundle

    def _load_bundle(self, algo: str, env: str) -> ArtifactBundle:
        if algo not in self.file_map:
            raise UnsupportedAlgorithmError(
                f"Алгоритм '{algo}' не поддержан. Доступные: {sorted(self.file_map.keys())}"
            )

        mapping = self.file_map[algo]
        missing_keys = [k for k in self._ARTIFACT_KEYS if k not in mapping]
        if missing_keys:
            raise UnsupportedAlgorithmError(
                f"Для algo='{algo}' в file_map не хватает ключей: {missing_keys}"
            )

        root_dir = self.models_root / algo / env
        if not root_dir.exists():
            raise ArtifactNotFoundError(
                f"Директория не найдена: {root_dir}. Проверьте models_root/algo/env."
            )

        paths: Dict[str, Path] = {k: (root_dir / mapping[k]) for k in self._ARTIFACT_KEYS}

        not_found = [str(p) for p in paths.values() if not p.exists()]
        if not_found:
            raise ArtifactNotFoundError(
                "Не найдены файлы артефактов:\n" + "\n".join(not_found)
            )

        # joblib.load умеет грузить sklearn/lightgbm-совместимые pkl
        model = joblib.load(paths["model"])
        label_encoder = joblib.load(paths["label_encoder"])
        feature_selector = joblib.load(paths["feature_selector"])
        numeric_imputer = joblib.load(paths["numeric_imputer"])
        categorical_imputer = joblib.load(paths["categorical_imputer"])

        return ArtifactBundle(
            model=model,
            label_encoder=label_encoder,
            feature_selector=feature_selector,
            numeric_imputer=numeric_imputer,
            categorical_imputer=categorical_imputer,
            root_dir=root_dir,
        )

    # ---------------------------
    # Data preparation
    # ---------------------------

    def _to_dataframe(self, data: InputData) -> "pd.DataFrame":
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, np.ndarray):
            # Без имён колонок мы всё равно можем работать, но импутация по dtype будет ограничена
            return pd.DataFrame(data)

        if isinstance(data, list):
            # list[dict]
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # dict может быть:
            # - row-like: {"f1": 1, "f2": 2}
            # - column-like: {"f1": [1,2], "f2": [3,4]}
            # Попробуем распознать по значениям
            if any(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in data.values()):
                return pd.DataFrame(data)
            return pd.DataFrame([data])

        raise ModelManagerError(f"Неподдерживаемый тип входных данных: {type(data)!r}")

    def _apply_imputation(
        self,
        X: "pd.DataFrame",
        numeric_imputer: Any,
        categorical_imputer: Any,
    ) -> "pd.DataFrame":
        X_out = X.copy()

        # Numeric
        num_cols = self._infer_columns_for_transformer(X_out, numeric_imputer, kind="numeric")
        if num_cols:
            X_out = self._apply_transformer_to_columns(X_out, numeric_imputer, num_cols)

        # Categorical
        cat_cols = self._infer_columns_for_transformer(X_out, categorical_imputer, kind="categorical")
        if cat_cols:
            X_out = self._apply_transformer_to_columns(X_out, categorical_imputer, cat_cols)

        return X_out

    def _infer_columns_for_transformer(self, X: "pd.DataFrame", transformer: Any, kind: str) -> List[str]:
        """
        Определяем столбцы для применения transformer.

        Приоритет:
        1) transformer.feature_names_in_ (если есть)
        2) Эвристика по dtype (numeric/categorical)
        """
        if transformer is None:
            return []

        cols: List[str] = []
        if hasattr(transformer, "feature_names_in_"):
            try:
                cols = [c for c in list(getattr(transformer, "feature_names_in_")) if c in X.columns]
                return cols
            except Exception:
                pass

        # fallback: dtype эвристика
        if kind == "numeric":
            cols = list(X.select_dtypes(include=[np.number]).columns)
        elif kind == "categorical":
            # object, string, category, bool часто идут как категориальные
            cols = list(X.select_dtypes(include=["object", "string", "category", "bool"]).columns)
        else:
            cols = []

        return cols

    def _apply_transformer_to_columns(self, X: "pd.DataFrame", transformer: Any, cols: List[str]) -> "pd.DataFrame":
        if transformer is None or not cols:
            return X

        if not hasattr(transformer, "transform"):
            raise ModelManagerError(f"Transformer {type(transformer)!r} не имеет метода transform().")

        # sklearn обычно возвращает ndarray; сохраним форму и вернём в DataFrame
        transformed = transformer.transform(X[cols])

        # Приведём к ndarray
        arr = np.asarray(transformed)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[1] != len(cols):
            # Например, OneHotEncoder расширяет пространство — но у нас imputer должен сохранять ширину.
            raise ModelManagerError(
                f"Неожиданная форма после transform для колонок {cols}: получено {arr.shape}, ожидалось (*, {len(cols)})."
            )

        X_out = X.copy()
        X_out.loc[:, cols] = arr
        return X_out

    def _apply_feature_selection(self, X: "pd.DataFrame", feature_selector: Any) -> Union["pd.DataFrame", "np.ndarray"]:
        if feature_selector is None:
            return X

        if not hasattr(feature_selector, "transform"):
            raise ModelManagerError(
                f"Feature selector {type(feature_selector)!r} не имеет метода transform()."
            )

        # Многие селекторы ожидают ndarray; но sklearn обычно принимает DataFrame тоже.
        selected = feature_selector.transform(X)

        # Если селектор вернул ndarray — оставим ndarray для модели (обычно безопаснее)
        if isinstance(selected, pd.DataFrame):
            return selected
        return np.asarray(selected)

    # ---------------------------
    # Prediction / decoding
    # ---------------------------

    def _model_predict(self, model: Any, X: Union["pd.DataFrame", "np.ndarray"]) -> Any:
        """
        Унифицированный вызов предсказания.

        Предпочитаем predict_proba если доступно, но пользователь явно просил использовать model.predict().
        Поэтому:
        - если у модели есть predict() — используем его
        - дальше нормализуем вывод (argmax, threshold и т.д.)
        """
        if not hasattr(model, "predict"):
            raise ModelManagerError(f"Модель {type(model)!r} не имеет метода predict().")

        return model.predict(X)

    def _normalize_predictions_to_class_indices(self, raw_pred: Any, label_encoder: Any) -> np.ndarray:
        """
        Приводит результат model.predict() к 1D массиву индексов классов (int).

        Поддерживает:
        - multiclass probabilities: shape (n, n_classes) -> argmax
        - multiclass logits/scores: shape (n, n_classes) -> argmax
        - binary probabilities: shape (n,) float in [0,1] -> threshold 0.5 -> {0,1}
        - already class indices: shape (n,) ints -> as-is
        """
        arr = np.asarray(raw_pred)

        # multiclass probabilities/scores
        if arr.ndim == 2:
            return np.argmax(arr, axis=1).astype(int)

        # single dimension
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        # If already strings/objects (редкий кейс) — попробуем вернуть как "индексы" через label_encoder
        # но лучше всё-таки трактовать как готовые метки: это обработаем позже в _decode_labels.
        if arr.dtype.kind in {"U", "S", "O"}:
            return arr  # type: ignore[return-value]

        # numeric 1D
        # binary probability heuristic
        n_classes = None
        if hasattr(label_encoder, "classes_"):
            try:
                n_classes = len(getattr(label_encoder, "classes_"))
            except Exception:
                n_classes = None

        if arr.dtype.kind in {"f"} and n_classes == 2:
            # Если значения похожи на вероятности, порог 0.5
            if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
                return (arr >= 0.5).astype(int)

        # otherwise treat as class indices
        return arr.astype(int)

    def _decode_labels(self, label_encoder: Any, class_indices: np.ndarray) -> List[str]:
        """
        Декодирует индексы классов в строковые метки.

        - Если class_indices уже строки/объекты, просто приводим к str.
        - Иначе используем label_encoder.inverse_transform().
        """
        arr = np.asarray(class_indices)

        if arr.dtype.kind in {"U", "S", "O"}:
            return [str(x) for x in arr.tolist()]

        if label_encoder is None or not hasattr(label_encoder, "inverse_transform"):
            raise ModelManagerError("label_encoder отсутствует или не имеет inverse_transform().")

        decoded = label_encoder.inverse_transform(arr.astype(int))
        # sklearn может вернуть ndarray любой dtype; приведём к list[str]
        return [str(x) for x in np.asarray(decoded).tolist()]


__all__ = [
    "ModelManager",
    "ModelManagerError",
    "UnsupportedAlgorithmError",
    "ArtifactNotFoundError",
]