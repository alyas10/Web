from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any,Callable
import os


class BaseDataLoader(ABC):
    """
    Базовый класс только для чтения данных из различных источников.
    Ответственность: превратить файл (csv, pcap, parquet) в 'сырой' DataFrame.
    """

    @abstractmethod
    def load(self, file_path: str, chunksize: int = 50000,
             progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        """
        Загружает файл и возвращает сырой DataFrame.

        :param file_path: Путь к файлу.
        :param chunksize: Размер порции (чанка) для итеративного чтения (важно для больших файлов).
        :param progress_callback: Функция обратного вызова, которая вызывается после обработки каждого чанка.
                                  Принимает один аргумент: количество обработанных строк.
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Список поддерживаемых расширений файлов (например, ['.csv', '.pcap'])."""
        pass


class DataPipelineAdapter:
    """
    Адаптер, который связывает сырые данные от загрузчика с ML-пайплайном.
     Выполняет подготовку данных с учётом настроек приложения:
    - нормализация числовых признаков
    - балансировка классов
    - разделение на train/test (если нужно)
    """

    def __init__(self, expected_features: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None):
        """
        :param expected_features: Список колонок, которые ожидает модель (из feature_info.pkl).
                                  Если None, проверка колонок пропускается (модель сама обработает).
        :param config: Словарь настроек из app_config.json или аналогичного источника.
                       Ожидаемые ключи:
                       - normalize_features (bool): применять ли нормализацию
                       - balance_classes (bool): применять ли балансировку классов
                       - auto_preprocess (bool): применять ли автоматическую предобработку
        """
        self.expected_features = expected_features
        self.config = config or {}

    def _get_config_value(self, key: str, default: Any = False) -> Any:
        """Безопасное получение значения из конфига."""
        return self.config.get(key, default)

    def prepare(self, raw_df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Выполняет подготовку данных с учётом настроек:
        1. Очистка от полностью пустых строк/колонок.
        2. Стандартизация имен колонок (lowercase, trim).
        3. Проверка наличия обязательных признаков (если заданы).
        4. Добавление отсутствующих колонок со значением 0 (для совместимости) - ОПТИМИЗИРОВАНО.
        5. Нормализация числовых признаков (если включено в настройках).
        6. Балансировка классов (если включено в настройках).

        :param raw_df: Исходный DataFrame
        :param target_column: Имя колонки с целевой переменной (для балансировки)
        :return: Подготовленный DataFrame
        """
        df = raw_df.copy()

        # 1. Базовая очистка
        df.dropna(how='all', inplace=True)
        if df.empty:
            raise ValueError("Файл не содержит корректных данных после очистки.")

        # 2. Нормализация имен колонок (убираем пробелы, нижний регистр)
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

        # 3. Логика согласования с моделью - ОПТИМИЗИРОВАНО (без цикла insert)
        if self.expected_features:
            # ОПТИМИЗАЦИЯ: Собираем все отсутствующие колонки в словарь
            missing_cols = [col for col in self.expected_features if col not in df.columns]

            if missing_cols:
                # ОПТИМИЗАЦИЯ: Создаем DataFrame с нулями для отсутствующих колонок
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
                # ОПТИМИЗАЦИЯ: Конкатенируем все сразу (одна операция вместо N insert)
                df = pd.concat([df, missing_df], axis=1)

            # Оставляем только нужные колонки в правильном порядке
            # Важно: порядок должен совпадать с обучением модели!
            # ОПТИМИЗАЦИЯ: Проверяем, что все колонки есть перед выборкой
            existing_cols = [col for col in self.expected_features if col in df.columns]
            if len(existing_cols) != len(self.expected_features):
                # Если всё ещё есть проблемы - заполняем недостающие
                for col in self.expected_features:
                    if col not in df.columns:
                        df[col] = 0
            df = df[self.expected_features]

        # 4. Применение настроек обработки данных
        if self._get_config_value('auto_preprocess', True):
            # Нормализация числовых признаков
            if self._get_config_value('normalize_features', False):
                df = self._normalize_features(df)

            # Балансировка классов (только если есть target_column)
            if self._get_config_value('balance_classes', False) and target_column:
                df = self._balance_classes(df, target_column)

        # ОПТИМИЗАЦИЯ: Дефрагментация DataFrame (убираем предупреждение PerformanceWarning)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            df = df.copy()

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет нормализацию к числовым колонкам.
        Использует Min-Max нормализацию [0, 1].
        ОПТИМИЗИРОВАНО: Векторизованная обработка вместо цикла.
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if len(numeric_cols) == 0:
            return df

        df_normalized = df.copy()

        # ОПТИМИЗАЦИЯ: Векторизованная нормализация всех колонок сразу
        min_vals = df[numeric_cols].min()
        max_vals = df[numeric_cols].max()
        ranges = max_vals - min_vals

        # Только колонки с ненулевым диапазоном
        valid_cols = ranges > 0
        if valid_cols.any():
            cols_to_norm = numeric_cols[valid_cols]
            df_normalized[cols_to_norm] = (df[cols_to_norm] - min_vals[cols_to_norm]) / ranges[cols_to_norm]

        # Колонки с нулевым диапазоном (все значения одинаковые) заполняем 0
        zero_range_cols = numeric_cols[~valid_cols]
        if len(zero_range_cols) > 0:
            df_normalized[zero_range_cols] = 0.0

        return df_normalized

    def _balance_classes(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Применяет балансировку классов методом downsampling.
        Уменьшает количество примеров в мажоритарных классах до уровня миноритарного.
        ОПТИМИЗИРОВАНО: Использует groupby apply для лучшей производительности.
        """
        if target_column not in df.columns:
            return df

        df_balanced = df.copy()

        # ОПТИМИЗАЦИЯ: Ранний выход, если балансировка не нужна
        class_counts = df_balanced[target_column].value_counts()

        # Если все классы уже сбалансированы (или один класс)
        if len(class_counts) <= 1 or class_counts.min() == class_counts.max():
            return df_balanced

        min_count = class_counts.min()

        if min_count == 0:
            return df_balanced

        # ОПТИМИЗАЦИЯ: Используем groupby + sample для эффективной балансировки
        balanced_dfs = []

        for class_label in class_counts.index:
            class_df = df_balanced[df_balanced[target_column] == class_label]
            if len(class_df) > min_count:
                # Downsampling - случайная выборка min_count примеров
                class_df = class_df.sample(n=min_count, random_state=42, replace=False)
            balanced_dfs.append(class_df)

        # ОПТИМИЗАЦИЯ: Однократная конкатенация и перемешивание
        df_balanced = pd.concat(balanced_dfs, axis=0, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        return df_balanced

    def train_test_split(
            self,
            df: pd.DataFrame,
            target_column: str,
            test_size: float = 0.2,
            random_state: int = 42,
            stratify: bool = True
    ) -> tuple:
        """
        Разделяет данные на обучающую и тестовую выборки.

        :param df: DataFrame с данными
        :param target_column: Имя колонки с целевой переменной
        :param test_size: Доля тестовой выборки (0.0 - 1.0)
        :param random_state: Seed для воспроизводимости
        :param stratify: Если True, сохраняет распределение классов
        :return: Кортеж (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            print("[WARNING] Scikit-learn не установлен, разделение данных пропущено.")
            return df, pd.DataFrame()

        if target_column not in df.columns:
            return df, pd.DataFrame()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        stratify_param = y if stratify else None

        # ОПТИМИЗАЦИЯ: Используем классическое разделение
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

    def validate_structure(self, df: pd.DataFrame) -> bool:
        """Быстрая проверка структуры перед тяжелой обработкой."""
        if self.expected_features is None:
            return True
        return all(col in df.columns for col in self.expected_features)

    # ДОБАВЛЕНЫ НОВЫЕ МЕТОДЫ ДЛЯ УЛУЧШЕНИЯ ПРЕДОБРАБОТКИ

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'zeros') -> pd.DataFrame:
        """
        Улучшенная обработка пропущенных значений.

        :param df: DataFrame для обработки
        :param strategy: Стратегия заполнения ('zeros', 'mean', 'median', 'mode', 'forward_fill')
        :return: DataFrame с заполненными пропусками
        """
        df_clean = df.copy()

        if strategy == 'zeros':
            df_clean = df_clean.fillna(0)
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0] if not df_clean.mode().empty else 0)
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

        return df_clean

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Удаление выбросов из числовых колонок.

        :param df: DataFrame для обработки
        :param method: Метод обнаружения выбросов ('iqr', 'zscore')
        :param threshold: Порог для определения выбросов
        :return: DataFrame без выбросов
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

        if method == 'iqr':
            # IQR метод
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Заменяем выбросы на границы вместо удаления
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore' and threshold > 0:
            # Z-score метод
            for col in numeric_cols:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean.loc[z_scores > threshold, col] = df_clean[col].mean()

        return df_clean

    def convert_categorical(self, df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Преобразование категориальных признаков с оптимизацией памяти.

        :param df: DataFrame для обработки
        :param categorical_cols: Список категориальных колонок (если None - автоопределение)
        :return: DataFrame с преобразованными категориями
        """
        df_conv = df.copy()

        if categorical_cols is None:
            # Автоопределение: строковые колонки и колонки с малым количеством уникальных значений
            for col in df_conv.columns:
                if df_conv[col].dtype == 'object' or df_conv[col].nunique() < df_conv.shape[0] * 0.05:
                    categorical_cols = (categorical_cols or []) + [col]

        if categorical_cols:
            for col in categorical_cols:
                if col in df_conv.columns:
                    # Используем category тип для экономии памяти
                    df_conv[col] = df_conv[col].astype('category')

        return df_conv

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Оптимизация типов данных для экономии памяти.

        :param df: DataFrame для оптимизации
        :return: DataFrame с оптимизированными типами
        """
        df_opt = df.copy()

        for col in df_opt.columns:
            col_type = df_opt[col].dtype

            if col_type != 'object':
                # Оптимизация числовых колонок
                c_min = df_opt[col].min()
                c_max = df_opt[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_opt[col] = df_opt[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_opt[col] = df_opt[col].astype(np.float32)
            else:
                # Оптимизация строковых колонок (категории)
                if df_opt[col].nunique() < df_opt.shape[0] * 0.5:
                    df_opt[col] = df_opt[col].astype('category')

        return df_opt