import pandas as pd
import numpy as np
import joblib

# --- Загрузка feature_info.pkl ---
# (Вам нужно распаковать его из бинарного дампа или использовать, если он у вас есть как файл)
# Примерный словарь, расшифрованный из вашего дампа:
# feature_info = {
#     'numeric_features': [...],
#     'categorical_features': [...],
#     'n_features_in': 229,
#     'classes_': [...],
# }
# Для этого скрипта предположим, что файл 'feature_info.pkl' уже существует и находится в 'models/lightgbm/test/'
# Или вы можете раскомментировать и использовать словарь выше, если он у вас есть в коде.

# Путь к файлу
FEATURE_INFO_PATH = "models/lightgbm/test/feature_info.pkl"

try:
    feature_info = joblib.load(FEATURE_INFO_PATH)
    print(f"✅ Загружена информация о признаках из {FEATURE_INFO_PATH}")
except FileNotFoundError:
    print(f"❌ Файл {FEATURE_INFO_PATH} не найден. Попробуем использовать данные из дампа...")
    # --- ВРУЧНУЮ РАСПАКОВАННЫЕ ДАННЫЕ ИЗ ВАШЕГО ДАМПА ---
    numeric_features = [
        'No.', 'Time', 'Length', 'Info', 'frame number', 'frame length', 'Frame Time', 'Frame Time (Epoch)',
        'Ethernet Source', 'Ethernet Destination', 'IP Length', 'IP TTL', 'IP Fragment Offset', 'IP Version',
        'IP DSCP Field', 'IP Checksum', 'TCP Source Port', 'TCP Destination Port', 'TCP Length',
        'TCP Sequence Number', 'TCP Acknowledgment Number', 'TCP SYN Flag', 'TCP ACK Flag', 'TCP FIN Flag',
        'TCP RST Flag', 'TCP Window Size', 'TCP Checksum', 'TCP Stream', 'UDP Source Port',
        'UDP Destination Port', 'UDP Length', 'UDP Checksum', 'ICMP Checksum', 'HTTP Request Method',
        'HTTP Request URI', 'HTTP Request Version', 'HTTP Full URI', 'HTTP Content-Length', 'HTTP Cookie',
        'HTTP Referer', 'HTTP Location', 'HTTP Authorization', 'HTTP Connection', 'DNS Query Type',
        'deltatime', 'Protocol_ARP', 'Protocol_BROWSER', 'Protocol_DHCP', 'Protocol_DHCPv6',
        'Protocol_DISTCC ', 'Protocol_DNS', 'Protocol_EXEC', 'Protocol_FTP', 'Protocol_HTTP',
        'Protocol_HTTP/XML', 'Protocol_ICMP', 'Protocol_ICMPv6', 'Protocol_IGMPv3', 'Protocol_IRC',
        'Protocol_LANMAN', 'Protocol_MDNS', 'Protocol_MySQL', 'Protocol_NBNS', 'Protocol_NBSS',
        'Protocol_NFS', 'Protocol_PGSQL', 'Protocol_Portmap', 'Protocol_RMI', 'Protocol_RPC',
        'Protocol_RSH', 'Protocol_RSTAT', 'Protocol_Rlogin', 'Protocol_SIP', 'Protocol_SMB',
        'Protocol_SMB2', 'Protocol_SMTP', 'Protocol_SSDP', 'Protocol_SSH', 'Protocol_SSHv1',
        'Protocol_SSHv2', 'Protocol_SSLv2', 'Protocol_SSLv3', 'Protocol_TCP', 'Protocol_TELNET',
        'Protocol_TLSv1', 'Protocol_TLSv1.2', 'Protocol_UDP', 'Protocol_VNC', 'Protocol_X11',
        'IP Protocol_ICMP,UDP', 'IP Protocol_IGMP', 'IP Protocol_TCP', 'IP Protocol_UDP',
        'IP Protocol_Unknown', 'HTTP Response Code_204.0', 'HTTP Response Code_301.0',
        'HTTP Response Code_302.0', 'HTTP Response Code_303.0', 'HTTP Response Code_304.0',
        'HTTP Response Code_401.0', 'HTTP Response Code_403.0', 'HTTP Response Code_404.0',
        'HTTP Response Code_500.0', 'HTTP Response Code_505.0', 'HTTP Response Code_NA',
        'TCP Flags_0x002', 'TCP Flags_0x004', 'TCP Flags_0x010', 'TCP Flags_0x011',
        'TCP Flags_0x012', 'TCP Flags_0x014', 'TCP Flags_0x018', 'TCP Flags_0x019',
        'TCP Flags_0x029', 'TCP Flags_0x02b', 'TCP Flags_0x8c2', 'IP Flags_0x00,0x00',
        'IP Flags_0x00,0x40', 'IP Flags_0x40', 'Ethernet Type_IPv4', 'Ethernet Type_IPv6',
        'ICMP Type_3.0', 'ICMP Type_9.0', 'Source_freq', 'Destination_freq', 'IP Source_freq',
        'IP Destination_freq', 'HTTP Host_freq', 'DNS Query Name_freq', 'HTTP User-Agent_freq',
        'HTTP Content Type_freq'
    ]
    categorical_features = [
        'Source', 'Destination', 'IP Source', 'IP Destination', 'HTTP User-Agent', 'HTTP Content Type',
        'HTTP Host', 'DNS Query Name', 'LastProto_0', 'LastProto_1', 'LastProto_10', 'LastProto_11',
        'LastProto_12', 'LastProto_13', 'LastProto_14', 'LastProto_15', 'LastProto_16', 'LastProto_17',
        'LastProto_18', 'LastProto_19', 'LastProto_2', 'LastProto_20', 'LastProto_21', 'LastProto_22',
        'LastProto_23', 'LastProto_24', 'LastProto_25', 'LastProto_26', 'LastProto_27', 'LastProto_28',
        'LastProto_29', 'LastProto_3', 'LastProto_30', 'LastProto_31', 'LastProto_32', 'LastProto_33',
        'LastProto_34', 'LastProto_35', 'LastProto_36', 'LastProto_37', 'LastProto_38', 'LastProto_39',
        'LastProto_4', 'LastProto_40', 'LastProto_41', 'LastProto_42', 'LastProto_43', 'LastProto_44',
        'LastProto_45', 'LastProto_46', 'LastProto_47', 'LastProto_48', 'LastProto_49', 'LastProto_5',
        'LastProto_50', 'LastProto_51', 'LastProto_52', 'LastProto_53', 'LastProto_54', 'LastProto_55',
        'LastProto_56', 'LastProto_57', 'LastProto_58', 'LastProto_59', 'LastProto_6', 'LastProto_60',
        'LastProto_61', 'LastProto_62', 'LastProto_63', 'LastProto_64', 'LastProto_65', 'LastProto_66',
        'LastProto_67', 'LastProto_68', 'LastProto_69', 'LastProto_7', 'LastProto_70', 'LastProto_71',
        'LastProto_72', 'LastProto_73', 'LastProto_74', 'LastProto_75', 'LastProto_76', 'LastProto_77',
        'LastProto_78', 'LastProto_79', 'LastProto_8', 'LastProto_80', 'LastProto_81', 'LastProto_82',
        'LastProto_83', 'LastProto_84', 'LastProto_85', 'LastProto_86', 'LastProto_87', 'LastProto_88',
        'LastProto_9', 'LastProto_nan'
    ]
    classes_ = ['ARP-spoof', 'Benign', 'FTP-Attack', 'Fuzzing']
    n_features_in = 229
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'n_features_in': n_features_in,
        'classes_': classes_,
    }
    print("✅ Использованы данные из дампа.")

all_features = feature_info['numeric_features'] + feature_info['categorical_features']
n_features = len(all_features)
print(f"📁 Найдено {n_features} признаков: {len(feature_info['numeric_features'])} числовых, {len(feature_info['categorical_features'])} категориальных.")

# --- Создание заглушки ---
# Создаем DataFrame с 1 строкой и 229 колонками
# Заполняем числовые колонки нулями (тип float64)
# Заполняем категориальные колонки значением -1 (тип float64, как обычно делает OrdinalEncoder для unknown_value)
dummy_row = {}

for col in all_features:
    if col in feature_info['numeric_features']:
        dummy_row[col] = 0.0
    elif col in feature_info['categorical_features']:
        dummy_row[col] = -1.0 # Представляет неизвестное категориальное значение

dummy_df = pd.DataFrame([dummy_row])

print(f"\n📋 Создана заглушка DataFrame:")
print(f"   - Форма: {dummy_df.shape}")
print(f"   - Колонки: {list(dummy_df.columns)}")
print(f"   - Типы данных:\n{dummy_df.dtypes}")
print(f"\n🔍 Пример первой строки:\n{dummy_df.iloc[0]}")

# --- Сохранение заглушки (опционально) ---
SAVE_PATH = "dummy_input_for_model.csv"
dummy_df.to_csv(SAVE_PATH, index=False)
print(f"\n💾 Заглушка сохранена в '{SAVE_PATH}'")

# --- Проверка совместимости с ожидаемым количеством признаков ---
if n_features == n_features_in:
    print(f"\n✅ Количество признаков в заглушке ({n_features}) совпадает с ожидаемым ({n_features_in}).")
else:
    print(f"\n❌ Количество признаков в заглушке ({n_features}) НЕ совпадает с ожидаемым ({n_features_in}).")
