# visualizer.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import pandas as pd


class DatasetVisualizer:
    """
    Класс для генерации визуализаций кратких характеристик датасета.
    """

    def __init__(self, feature_info):
        self.feature_info = feature_info
        self.numeric_features = set(feature_info['numeric_features'])
        self.categorical_features = set(feature_info['categorical_features'])

    def _create_card(self, title, figure_html):
        """
        Вспомогательный метод для создания HTML-карточки.
        """
        card_html = f"""
        <div class="visualization-card card">
            <div class="card-header">
                <h4 class="card-title">{title}</h4>
            </div>
            <div class="card-content">
                {figure_html}
            </div>
        </div>
        """
        return card_html

    def generate_overview_plots(self, df):
        """
        Генерирует основные графики для обзора датасета.
        Возвращает список HTML-строк для карточек.
        """
        plots_html = []

        # --- 1. Гистограмма протоколов ---
        # Ищем колонки, связанные с протоколами (например, Protocol_TCP, Protocol_UDP)
        protocol_cols = [col for col in df.columns if col.startswith('Protocol_') and col in self.categorical_features]
        if protocol_cols:
            # Суммируем единицы для каждого протокола (one-hot encoded)
            protocol_counts = df[protocol_cols].sum().sort_values(ascending=False)
            if not protocol_counts.empty:
                fig_protocol = px.bar(
                    x=protocol_counts.index,
                    y=protocol_counts.values,
                    title="Распределение протоколов",
                    labels={'x': 'Протокол', 'y': 'Количество'},
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_protocol.update_layout(height=400)
                plot_div_protocol = fig_protocol.to_html(full_html=False, include_plotlyjs=False)
                plots_html.append(self._create_card("Распределение протоколов", plot_div_protocol))

        # --- 2. Гистограмма IP-адресов (Top N) ---
        # Ищем колонки, связанные с IP (Source, Destination, etc.)
        ip_cols = [col for col in df.columns if 'ip' in col.lower() and col in self.categorical_features]
        combined_ips = pd.Series(dtype='object')
        for col in ip_cols:
            if col in df.columns:
                series = df[col].dropna()
                combined_ips = pd.concat([combined_ips, series])

        if not combined_ips.empty:
            top_ips = combined_ips.value_counts().head(10)
            fig_ip = px.bar(
                x=top_ips.index,
                y=top_ips.values,
                title="Топ-10 IP-адресов",
                labels={'x': 'IP-адрес', 'y': 'Количество'},
                color_discrete_sequence=px.colors.sequential.Viridis_r
            )
            fig_ip.update_layout(height=400)
            plot_div_ip = fig_ip.to_html(full_html=False, include_plotlyjs=False)
            plots_html.append(self._create_card("Топ-10 IP-адресов", plot_div_ip))

        # --- 3. Круговая диаграмма для метки (если есть) ---
        # Поиск колонки метки (label/class)
        label_col = None
        possible_label_names = ['label', 'class', 'Label', 'Class']
        for col_name in possible_label_names:
            if col_name in df.columns:
                label_col = col_name
                break

        if label_col and label_col in df.columns:
            label_counts = df[label_col].value_counts()
            if not label_counts.empty:
                fig_label = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Распределение меток (Label)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_label.update_layout(height=400)
                plot_div_label = fig_label.to_html(full_html=False, include_plotlyjs=False)
                plots_html.append(self._create_card("Распределение меток", plot_div_label))

        # --- 4. Гистограмма длин пакетов (если есть) ---
        length_col_candidates = ['Length', 'length', 'Frame Time (Epoch)', 'frame_length', 'deltatime']
        length_col = None
        for col_name in length_col_candidates:
            if col_name in df.columns and col_name in self.numeric_features:
                length_col = col_name
                break

        if length_col:
            # Убираем NaN для гистограммы
            lengths = df[length_col].dropna()
            if not lengths.empty:
                fig_length = px.histogram(
                    x=lengths,
                    nbins=50,
                    title=f"Распределение по {length_col}",
                    labels={'x': length_col, 'y': 'Частота'},
                    color_discrete_sequence=px.colors.diverging.Portland
                )
                fig_length.update_layout(height=400)
                plot_div_length = fig_length.to_html(full_html=False, include_plotlyjs=False)
                plots_html.append(self._create_card(f"Распределение по {length_col}", plot_div_length))

        # --- 5. Статистика пропусков ---
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            missing_counts_filtered = missing_counts[missing_counts > 0]
            if not missing_counts_filtered.empty:
                fig_missing = px.bar(
                    x=missing_counts_filtered.index,
                    y=missing_counts_filtered.values,
                    title="Количество пропущенных значений по признакам",
                    labels={'x': 'Признак', 'y': 'Количество пропусков'},
                    color_discrete_sequence=px.colors.sequential.Magenta_r
                )
                fig_missing.update_layout(height=400)
                plot_div_missing = fig_missing.to_html(full_html=False, include_plotlyjs=False)
                plots_html.append(self._create_card("Пропуски в данных", plot_div_missing))

        return plots_html
