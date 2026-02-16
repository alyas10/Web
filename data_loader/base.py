from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict, Any
import os


class BaseDataLoader(ABC):
    """
    Базовый класс только для чтения данных из различных источников.
    Ответственность: превратить файл (csv, pcap, parquet) в 'сырой' DataFrame.
    """

    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """Загружает файл и возвращает сырой DataFrame."""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Список поддерживаемых расширений файлов (например, ['.csv', '.pcap'])."""
        pass


class DataPipelineAdapter:
    """
    Адаптер, который связывает сырые данные от загрузчика с ML-пайплайном.
    Выполняет минимально необходимую подготовку: переименование колонок, 
    приведение типов и базовую очистку перед подачей в sklearn pipeline.
    """

    def __init__(self, expected_features: Optional[List[str]] = None):
        """
        :param expected_features: Список колонок, которые ожидает модель (из feature_info.pkl).
                                  Если None, проверка колонок пропускается (модель сама обработает).
        """
        self.expected_features = expected_features

    def prepare(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет подготовку данных:
        1. Очистка от полностью пустых строк/колонок.
        2. Стандартизация имен колонок (lowercase, trim).
        3. Проверка наличия обязательных признаков (если заданы).
        4. Добавление отсутствующих колонок со значением 0 (для совместимости).
        """
        df = raw_df.copy()

        # 1. Базовая очистка
        df.dropna(how='all', inplace=True)
        if df.empty:
            raise ValueError("Файл не содержит корректных данных после очистки.")

        # 2. Нормализация имен колонок (убираем пробелы, нижний регистр)
        #df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

        # 3. Логика согласования с моделью
        if self.expected_features:
            # Проверяем, чего не хватает
            for col in self.expected_features:
                if col not in df.columns:
                    df[col] = 0  # Заполняем только реально отсутствующие


            # Оставляем только нужные колонки в правильном порядке
            # Важно: порядок должен совпадать с обучением модели!
            df = df[self.expected_features]

        return df

    def validate_structure(self, df: pd.DataFrame) -> bool:
        """Быстрая проверка структуры перед тяжелой обработкой."""
        if self.expected_features is None:
            return True
        return all(col in df.columns for col in self.expected_features)