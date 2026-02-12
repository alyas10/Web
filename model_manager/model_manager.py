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


class ModelManager:
    def __init__(self, models_root: Optional[Union[str, Path]] = None):
        self._lock = RLock()
        self._cache = {}
        self.models_root = Path(models_root or Path(__file__).parent.parent / "models")

        self.file_map = {
            "lightgbm": {
                "pipeline": "full_pipeline.pkl",
                "label_encoder": "label_encoder_full_dataset.pkl",
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
                raise FileNotFoundError(p)

        pipeline = joblib.load(paths["pipeline"])
        label_encoder = joblib.load(paths["label_encoder"])

        return ArtifactBundle(
            pipeline=pipeline,
            label_encoder=label_encoder,
            root_dir=root_dir,
        )

    def predict(self, algo: str, data: InputData, env: str = "test") -> List[str]:
        bundle = self._get_or_load_bundle(algo, env)

        if isinstance(data, np.ndarray):
            # Создайте DataFrame с правильными колонками (если вы знаете их)
            # Для теста — используем числа, но без имён
            data = pd.DataFrame(data)

        try:
            pred = bundle.pipeline.predict(data)
            print(
                f"[DEBUG] pred type: {type(pred)}, shape: {getattr(pred, 'shape', 'N/A')}, value: {pred[:5] if hasattr(pred, '__len__') else 'N/A'}")
        except Exception as e:
            print(f"[ERROR] pipeline.predict failed: {e}")
            raise

        if pred is None:
            raise ModelManagerError("Pipeline.predict() returned None")

        try:
            labels = bundle.label_encoder.inverse_transform(pred.astype(int))
        except Exception as e:
            print(f"[ERROR] inverse_transform failed: {e}")
            raise

        return [str(x) for x in labels]