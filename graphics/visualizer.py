# visualizer.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import pandas as pd
import networkx as nx
from collections import Counter


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
        protocol_cols = [col for col in df.columns if col.startswith('Protocol_') and col in self.categorical_features]
        if protocol_cols:
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

        # --- 6. Дополнительное распределение протоколов (Pie) ---
        protocol_raw_col = None
        for col in ['proto', 'protocol', 'Proto', 'Protocol']:
            if col in df.columns:
                protocol_raw_col = col
                break

        if protocol_raw_col:
            protocol_counts = df[protocol_raw_col].value_counts()
            if not protocol_counts.empty:
                fig_pie = px.pie(
                    values=protocol_counts.values,
                    names=protocol_counts.index.astype(str),
                    title="🔗 Распределение протоколов",
                    color_discrete_sequence=px.colors.sequential.Blues,
                    template='plotly_white'
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False, height=400)
                plot_div = fig_pie.to_html(full_html=False, include_plotlyjs=False)
                plots_html.append(self._create_card("Распределение протоколов (Pie)", plot_div))

        # --- 7. Интенсивность трафика во времени ---
        timestamp_col = None
        for col in ['timestamp', 'time', 'Timestamp', 'Time']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            try:
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp[timestamp_col], errors='coerce')
                df_temp = df_temp.dropna(subset=['timestamp'])

                if not df_temp.empty:
                    df_temp['minute'] = df_temp['timestamp'].dt.floor('T')
                    traffic_by_time = df_temp.groupby('minute').size().reset_index(name='packets')

                    if not traffic_by_time.empty:
                        fig_line = px.line(
                            traffic_by_time,
                            x='minute',
                            y='packets',
                            title="📈 Интенсивность трафика во времени",
                            markers=True,
                            line_shape='spline',
                            color_discrete_sequence=['#0066cc'],
                            template='plotly_white'
                        )
                        fig_line.update_layout(height=400)
                        plot_div = fig_line.to_html(full_html=False, include_plotlyjs=False)
                        plots_html.append(self._create_card("Интенсивность трафика во времени", plot_div))
            except Exception:
                # Ошибка обработки времени игнорируется для стабильности визуализации
                pass

        return plots_html

    def generate_data_summary(self, df):
        """
        Генерирует краткую характеристику датасета в виде словаря.
        Включает: количество строк/колонок, распределение протоколов, IP-адресов,
        а также статистику графа связей (networkx).
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'protocols': {},
            'top_src_ips': {},
            'top_dst_ips': {},
            'graph_stats': {},
            'label_distribution': {}
        }

        # 1. Распределение протоколов
        protocol_col = None
        for col in ['proto', 'protocol', 'Proto', 'Protocol']:
            if col in df.columns:
                protocol_col = col
                break

        if protocol_col:
            protocol_counts = df[protocol_col].value_counts().to_dict()
            summary['protocols'] = {str(k): int(v) for k, v in protocol_counts.items()}

        # 2. Топ IP-адресов (источники и назначения)
        for ip_type, col_patterns in [('src', ['src_ip', 'source_ip', 'ip_src']),
                                      ('dst', ['dst_ip', 'destination_ip', 'ip_dst'])]:
            ip_col = None
            for pattern in col_patterns:
                if pattern in df.columns:
                    ip_col = pattern
                    break

            if ip_col:
                ip_counts = df[ip_col].dropna().value_counts().head(10).to_dict()
                summary[f'top_{ip_type}_ips'] = {str(k): int(v) for k, v in ip_counts.items()}

        # 3. Граф связей (networkx) - строим граф src_ip -> dst_ip
        src_ip_col = None
        dst_ip_col = None

        for pattern in ['src_ip', 'source_ip', 'ip_src']:
            if pattern in df.columns:
                src_ip_col = pattern
                break

        for pattern in ['dst_ip', 'destination_ip', 'ip_dst']:
            if pattern in df.columns:
                dst_ip_col = pattern
                break

        if src_ip_col and dst_ip_col:
            G = nx.DiGraph()

            # Добавляем рёбра из DataFrame
            edges = df[[src_ip_col, dst_ip_col]].dropna().values.tolist()
            G.add_edges_from(edges)

            # Статистика графа
            summary['graph_stats'] = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'avg_degree': round(sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1), 2),
                'is_connected': nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
                'num_components': nx.number_weakly_connected_components(G) if G.number_of_nodes() > 0 else 0
            }

            # Топ узлов по степени (количеству соединений)
            if G.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(G)
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                summary['graph_stats']['top_nodes'] = {str(k): round(v, 4) for k, v in top_nodes}

        # 4. Распределение меток (если есть)
        label_col = None
        for col_name in ['label', 'class', 'Label', 'Class']:
            if col_name in df.columns:
                label_col = col_name
                break

        if label_col:
            label_counts = df[label_col].value_counts().to_dict()
            summary['label_distribution'] = {str(k): int(v) for k, v in label_counts.items()}

        return summary
