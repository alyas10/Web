import sys, os, threading, time, socket
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from PySide6.QtGui import QIcon

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def wait_port(host, port, timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False

try:
    from app import create_app
    flask_app = create_app()
except Exception as e:
    print(f"Ошибка: не удалось создать Flask app: {e}")
    sys.exit(1)

def run_flask():
    flask_app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

def main():
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    if not wait_port("127.0.0.1", 5000, timeout=10):
        print("Flask не запустился за 10 секунд")
        sys.exit(1)

    app = QApplication(sys.argv)

    icon_path = resource_path(os.path.join("static", "icon.png"))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    app.setApplicationName("ML Network Security")

    window = QWebEngineView()
    window.setWindowTitle("ML Network Security Simulator")
    window.resize(1280, 800)
    window.setUrl(QUrl("http://127.0.0.1:5000/settings"))
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()