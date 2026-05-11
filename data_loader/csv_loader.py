from .base import BaseDataLoader
import pandas as pd
from typing import Optional, Callable


class CSVDataLoader(BaseDataLoader):
    @property
    def supported_extensions(self):
        return ['.csv']

    def load(self, file_path: str, chunksize: int = 50000,
             progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        """
        Загружает CSV файл по частям (чанками) для экономии памяти и отслеживания прогресса.
        """
        try:
            # Создаем итератор для чтения файла по chunksize строк за раз
            reader = pd.read_csv(file_path, low_memory=False, chunksize=chunksize)

            chunks = []
            rows_processed = 0

            # Итерируемся по чанкам
            for chunk in reader:
                chunks.append(chunk)
                rows_processed += len(chunk)

                # Если передана функция прогресса, вызываем её
                # Это позволит фронтенду или контроллеру обновить % загрузки
                if progress_callback:
                    progress_callback(rows_processed)

            # Если файл пустой
            if not chunks:
                return pd.DataFrame()

            # Объединяем все чанки в один DataFrame
            # ignore_index=True сбрасывает индексы, чтобы они шли 0, 1, 2...
            df = pd.concat(chunks, ignore_index=True)

            return df

        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {e}")