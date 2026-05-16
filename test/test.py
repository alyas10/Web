import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "pipeline"

print(f"Current Working Directory: {os.getcwd()}")
print(f"Resolved Models Root: {MODELS_ROOT}")
print(f"Exists: {MODELS_ROOT.exists()}")

print("\n Тестовая загрузка через ModelManager...")
try:
    from model_manager.model_manager import ModelManager

    mm = ModelManager(models_root=str(MODELS_ROOT))

    if 'random_forest' not in mm.file_map:
        print(" ОШИБКА: random_forest не найден в file_map ModelManager")

    bundle = mm._get_or_load_bundle('random_forest', 'test')
    print("    Bundle загружен успешно")
    print(f"   Тип pipeline: {type(bundle.pipeline).__name__}")
    print(f"   Тип label_encoder: {type(bundle.label_encoder).__name__}")

    meta_path = MODELS_ROOT / "random_forest" / "test" / "model_metadata.joblib"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        feature_names = meta.get('feature_names', [])
    else:
        txt_path = MODELS_ROOT / "random_forest" / "test" / "feature_names.txt"
        feature_names = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]

    if not feature_names:
        raise ValueError("feature_names не найдены в метаданных или файле")

    X_test = pd.DataFrame(
        np.random.randn(3, len(feature_names)),
        columns=feature_names
    )

    # Менеджер теперь сам обрабатывает тип предсказаний
    predictions = mm.predict('random_forest', X_test, env='test')
    print(f"  Предсказания: {predictions[:3]}")

    print(f"\nМодель готова к использованию в веб-интерфейсе.")
    print(f"ID модели для использования: 'random_forest'")

except Exception as e:
    print(f"\n ОШИБКА при тестировании: {e}")
    import traceback
    traceback.print_exc()