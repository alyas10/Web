from __future__ import annotations

from typing import List, Optional, Dict, Any
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

    def __init__(self, max_packets: Optional[int] = None, payload_preview_bytes: int = 0):
        """
        :param max_packets: если задано — ограничивает число пакетов для чтения (ускоряет).
        :param payload_preview_bytes: если > 0 — добавляет preview первых N байт Raw payload (как bytes).
        """
        self.max_packets = max_packets
        self.payload_preview_bytes = payload_preview_bytes

    @property
    def supported_extensions(self) -> List[str]:
        return [".pcap", ".pcapng"]

    def load(self, file_path: str) -> pd.DataFrame:
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

        df = pd.DataFrame(rows)

        # минимально полезно: сортировка по времени, если есть
        if not df.empty and "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True, ignore_index=True)

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