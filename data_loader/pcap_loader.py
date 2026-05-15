from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import os

import pandas as pd

from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        pass


class PcapScapyDataLoader(BaseDataLoader):
    """
    DataLoader для PCAP/PCAPNG на базе scapy.
    Возвращает "сырой" DataFrame со сведениями о пакетах.
    """

    def __init__(self, max_packets: Optional[int] = None, payload_preview_bytes: int = 0,
                 extract_features: bool = True):
        """
        :param max_packets: если задано — ограничивает число пакетов для чтения (ускоряет).
        :param payload_preview_bytes: если > 0 — добавляет preview первых N байт Raw payload (как bytes).
        :param extract_features: если True — извлекает дополнительные признаки для ML
        """
        self.max_packets = max_packets
        self.payload_preview_bytes = payload_preview_bytes
        self.extract_features = extract_features

    @property
    def supported_extensions(self) -> List[str]:
        return [".pcap", ".pcapng"]

    def load(self, file_path: str, chunksize: int = 10000,
             progress_callback: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
        # 1) Проверки пути/расширения
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path должен быть непустой строкой.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"Неподдерживаемое расширение '{ext}'. "
                f"Поддерживаются: {self.supported_extensions}"
            )

        # 2) Импорт scapy (здесь, чтобы модуль можно было импортировать без scapy при необходимости)
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP, ARP, DNS, Raw  # type: ignore
            # HTTP слои могут отсутствовать в зависимости от версии/пакета scapy
            try:
                from scapy.layers.http import HTTPRequest, HTTPResponse  # type: ignore
            except Exception:
                HTTPRequest, HTTPResponse = None, None  # noqa
        except ImportError as e:
            raise ImportError(
                "Scapy не установлен. Установите: pip install scapy"
            ) from e

        packets = rdpcap(file_path)

        rows: List[Dict[str, Any]] = []
        limit = self.max_packets if self.max_packets is not None else len(packets)

        for i, pkt in enumerate(packets[:limit]):
            row = self._parse_packet(
                pkt,
                packet_id=i,
                IP=IP, TCP=TCP, UDP=UDP, ICMP=ICMP, ARP=ARP, DNS=DNS, Raw=Raw,
                HTTPRequest=HTTPRequest, HTTPResponse=HTTPResponse,
            )
            if row is not None:
                rows.append(row)

             # Обновляем прогресс
            if progress_callback and (i + 1) % chunksize == 0:
                progress_callback(i + 1)

        df = pd.DataFrame(rows)

        # минимально полезно: сортировка по времени, если есть
        if not df.empty and "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True, ignore_index=True)

        # Извлечение дополнительных признаков для ML
        if self.extract_features and not df.empty:
            df = self._extract_ml_features(df)

        return df

    def _parse_packet(
        self,
        packet,
        packet_id: int,
        IP, TCP, UDP, ICMP, ARP, DNS, Raw,
        HTTPRequest=None, HTTPResponse=None
    ) -> Optional[Dict[str, Any]]:
        try:
            ts = None
            try:
                ts = datetime.fromtimestamp(float(packet.time))
            except Exception:
                ts = None

            info = ""
            protocol = "Unknown"
            src_ip = dst_ip = None
            src_port = dst_port = None

            if IP in packet:
                ip_layer = packet[IP]
                src_ip = getattr(ip_layer, "src", None)
                dst_ip = getattr(ip_layer, "dst", None)
                protocol = self._get_protocol_name(getattr(ip_layer, "proto", None))

                if TCP in packet:
                    tcp = packet[TCP]
                    protocol = "TCP"
                    src_port = int(getattr(tcp, "sport", 0)) if getattr(tcp, "sport", None) is not None else None
                    dst_port = int(getattr(tcp, "dport", 0)) if getattr(tcp, "dport", None) is not None else None

                    # HTTP (если доступно)
                    if HTTPRequest is not None and HTTPRequest in packet:
                        http_req = packet[HTTPRequest]
                        method = getattr(http_req, "Method", b"")
                        path = getattr(http_req, "Path", b"")
                        protocol = "HTTP"
                        info = f"HTTP Request: {method.decode(errors='ignore')} {path.decode(errors='ignore')}"
                    elif HTTPResponse is not None and HTTPResponse in packet:
                        http_resp = packet[HTTPResponse]
                        code = getattr(http_resp, "Status_Code", b"")
                        protocol = "HTTP"
                        info = f"HTTP Response: {code.decode(errors='ignore') if isinstance(code, (bytes, bytearray)) else code}"
                    else:
                        flags_repr = ""
                        try:
                            flags_repr = str(getattr(tcp, "flags", ""))
                        except Exception:
                            flags_repr = ""
                        seq = getattr(tcp, "seq", None)
                        info = f"TCP Flags={flags_repr} Seq={seq}"

                elif UDP in packet:
                    udp = packet[UDP]
                    protocol = "UDP"
                    src_port = int(getattr(udp, "sport", 0)) if getattr(udp, "sport", None) is not None else None
                    dst_port = int(getattr(udp, "dport", 0)) if getattr(udp, "dport", None) is not None else None

                    if DNS in packet:
                        dns = packet[DNS]
                        protocol = "DNS"
                        # запрос/ответ
                        qd = getattr(dns, "qd", None)
                        an = getattr(dns, "an", None)
                        if qd is not None and getattr(qd, "qname", None) is not None:
                            qname = qd.qname
                            info = f"DNS Query: {qname.decode(errors='ignore') if isinstance(qname, (bytes, bytearray)) else qname}"
                        elif an is not None:
                            # an может быть списком/слоем — безопасно показываем что есть
                            info = "DNS Response"
                    else:
                        length = getattr(udp, "len", None)
                        info = f"UDP Length={length}"

                elif ICMP in packet:
                    icmp = packet[ICMP]
                    protocol = "ICMP"
                    icmp_type = getattr(icmp, "type", None)
                    icmp_code = getattr(icmp, "code", None)
                    info = f"ICMP Type={icmp_type} Code={icmp_code}"

            elif ARP in packet:
                arp = packet[ARP]
                protocol = "ARP"
                src_ip = getattr(arp, "psrc", None)
                dst_ip = getattr(arp, "pdst", None)
                op = getattr(arp, "op", None)
                info = f"ARP {'Request' if op == 1 else 'Reply' if op == 2 else op}"

            raw_preview = None
            if self.payload_preview_bytes and Raw in packet:
                try:
                    raw_preview = bytes(packet[Raw])[: self.payload_preview_bytes]
                except Exception:
                    raw_preview = None

            return {
                "id": packet_id,
                "timestamp": ts,
                "length": int(len(packet)) if packet is not None else None,
                "protocol": protocol,
                "source_ip": src_ip,
                "destination_ip": dst_ip,
                "source_port": src_port,
                "destination_port": dst_port,
                "info": info,
                "raw_preview": raw_preview,
            }

        except Exception:
            # На проде лучше логировать, но не валить весь файл из-за одного пакета
            return None

    def _get_protocol_name(self, proto_num: Optional[int]) -> str:
        protocol_map = {
            1: "ICMP",
            6: "TCP",
            17: "UDP",
            2: "IGMP",
            47: "GRE",
            50: "ESP",
            51: "AH",
        }
        if proto_num is None:
            return "Unknown"
        return protocol_map.get(int(proto_num), f"Proto-{proto_num}")

    def _extract_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлекает дополнительные признаки из PCAP данных для ML моделей.
        Добавляет числовые признаки на основе сетевой активности.

        :param df: DataFrame с распарсенными пакетами
        :return: DataFrame с дополнительными признаками
        """
        try:
            # 1. Длительность пакетов (если есть timestamp)
            if 'timestamp' in df.columns and len(df) > 1:
                df['packet_interval'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

            # 2. Бинарные флаги для протоколов
            for proto in ['TCP', 'UDP', 'ICMP', 'DNS', 'HTTP', 'ARP']:
                df[f'is_{proto.lower()}'] = (df['protocol'] == proto).astype(int)

            # 3. Размер пакета (нормализованный)
            if 'length' in df.columns:
                df['length_normalized'] = df['length'] / df['length'].max() if df['length'].max() > 0 else 0

            # 4. Наличие портов (для TCP/UDP)
            df['has_source_port'] = df['source_port'].notna().astype(int)
            df['has_destination_port'] = df['destination_port'].notna().astype(int)

            # 5. Частота IP адресов (сколько раз встречался каждый IP)
            if 'source_ip' in df.columns:
                src_ip_counts = df.groupby('source_ip').size().to_dict()
                df['src_ip_frequency'] = df['source_ip'].map(src_ip_counts).fillna(1)

            if 'destination_ip' in df.columns:
                dst_ip_counts = df.groupby('destination_ip').size().to_dict()
                df['dst_ip_frequency'] = df['destination_ip'].map(dst_ip_counts).fillna(1)

            # 6. Известные порты (веб, DNS, SSH и т.д.)
            common_ports = {80: 'http', 443: 'https', 53: 'dns', 22: 'ssh', 21: 'ftp',
                            25: 'smtp', 110: 'pop3', 143: 'imap', 3389: 'rdp'}

            df['is_common_port'] = df['destination_port'].apply(
                lambda x: 1 if x in common_ports.keys() else 0
            ) if 'destination_port' in df.columns else 0

        except Exception as e:
            # Если извлечение признаков не удалось - продолжаем без них
            print(f"[WARNING] Не удалось извлечь ML признаки: {e}")

        return df
