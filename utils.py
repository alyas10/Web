# utils.py
def _label_color(label: str) -> str:
    l = (label or "").strip().lower()
    if l == "benign":
        return "green"
    if "dos" in l or "ddos" in l:
        return "red"
    if "intrusion" in l:
        return "orange"
    if "anomaly" in l:
        return "yellow"
    return "gray"