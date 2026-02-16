from .base import BaseDataLoader
import pandas as pd


class CSVDataLoader(BaseDataLoader):
    @property
    def supported_extensions(self):
        return ['.csv']

    def load(self, file_path: str) -> pd.DataFrame:
        # Автоматическое определение разделителя часто полезно для сетевых логов
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {e}")

        return df