# data_loader/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class DataProcessor(ABC):
    """
    Абстрактный базовый класс для загрузки и обработки сетевых данных.

    Любой формат (CSV, PCAP, Parquet и др.) или датасет (LSNM2024, NSL-KDD и др.)
    должен реализовать:
    - загрузку сырых данных,
    - извлечение признаков в единый формат, совместимый с ML-моделью.
    """

    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Загружает сырые данные из файла и возвращает их в виде DataFrame.

        Parameters:
            file_path (str): Путь к файлу.

        Returns:
            pd.DataFrame: Сырые данные (структура зависит от формата/датасета).
        """
        pass

    @abstractmethod
    def extract_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует сырые данные в набор признаков, совместимых с обученной моделью.
        Для LSNM2024 это 60 числовых/категориальных признаков на flow.
        Для PCAP — тоже 60 признаков, но вычисленных из пакетов.
        Для других CSV — может потребоваться адаптация (mapping колонок, нормализация).

        Parameters:
            raw_data (pd.DataFrame): Сырые данные после load().

        Returns:
            pd.DataFrame: Таблица с признаками (столбцы = features, строки = flows/instances).
                         Порядок и названия колонок должны соответствовать модели.
        """
        pass

    def process(self, file_path: str) -> pd.DataFrame:
        """
        Выполняет полный pipeline обработки:
        1. Загрузка данных,
        2. Извлечение признаков.

        Parameters:
            file_path (str): Путь к файлу данных.

        Returns:
            pd.DataFrame: Готовый датасет для подачи в модель.
        """
        raw_data = self.load(file_path)
        features = self.extract_features(raw_data)
        return features

    @abstractmethod
    def validate_compatibility(self, df: pd.DataFrame) -> bool:
        """
        Проверяет, совместимы ли признаки в df с ожидаемым форматом модели.
        Используется для проверки корректности extract_features().

        Parameters:
            df (pd.DataFrame): DataFrame после extract_features().

        Returns:
            bool: True, если совместим.
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> Optional[list]:
        """
        Возвращает список колонок, требуемых для модели.
        Может быть None, если неизвестен заранее (например, для PCAP).
        """
        pass