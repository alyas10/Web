import sys

from flask import Flask
import os
import json
from modules_site.dashboard import bp as dashboard_bp
from modules_site.data import bp as data_bp
from modules_site.models import bp as models_bp
from modules_site.results import bp as results_bp
from modules_site.settings import bp as settings_bp
from modules_site.dataset_analys import bp as dataset_analys_bp


from model_manager.model_utils import NumericFeatureSelector
from modules_site.settings.routes import load_config

def get_resource_path(relative_path):
    """Получить абсолютный путь к ресурсу"""
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def create_app():

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    # Загрузка настроек и применение к конфигу приложения
    app_config = load_config()
    app.config['MAX_CONTENT_LENGTH'] = app_config.get('max_file_size_mb', 1000) * 1024 * 1024
    app.app_config = app_config

    @app.context_processor
    def inject_app_config():
        """Делает app_config доступным во всех Jinja2-шаблонах"""
        return {
            'app_config': app.app_config,
            'project_name': app.app_config.get('project_name', 'ML Security Project'),
            'project_description': app.app_config.get('project_description', '')
        }

    # Очистка папки uploads при старте
    import os, shutil
    folder = app.config['UPLOAD_FOLDER']
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print("Папка uploads очищена.")
    except Exception as e:
        print(f" Ошибка очистки папки: {e}")

    # Загрузка feature_info и инициализация shared_objects
    import joblib
    try:
        pipe = joblib.load("pipeline/lightgbm/test/lgbm_optuna_best.joblib")
        features_path = "pipeline/lightgbm/test/feature_names.txt"
        with open(features_path, 'r', encoding='utf-8') as f:
            all_features = [line.strip() for line in f if line.strip()]
        feature_info = {
            'numeric_features': all_features,
            'categorical_features': []
        }
        print(f"Загружено {len(all_features)} признаков из feature_names.txt.")
    except Exception as e:
        print(f"Ошибка загрузки модели/признаков: {e}")
        feature_info = {'numeric_features': [], 'categorical_features': []}
        pipe = None

    # Загрузка настроек из app_config.json
    def load_app_config():
        config_path = os.path.join(os.path.dirname(__file__), 'app_config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # Возвращаем дефолтные значения
            return {
                "project_name": "ML Network Security Project",
                "normalize_features": True,
                "balance_classes": True,
                "auto_preprocess": True
            }

    import json
    app_config = load_app_config()

    # Инициализация общих объектов и передача их в blueprints
    from model_manager.model_manager import ModelManager
    from data_loader.base import DataPipelineAdapter
    from graphics.visualizer import DatasetVisualizer

    model_manager = ModelManager(models_root="pipeline")
    # Передаём конфиг в адаптер для использования настроек обработки данных
    data_adapter = DataPipelineAdapter(
        expected_features=feature_info['numeric_features'] + feature_info['categorical_features'],
        config=app_config,
        numeric_features=feature_info['numeric_features'],
        categorical_features=feature_info['categorical_features']
    )
    visualizer = DatasetVisualizer(feature_info=feature_info)

    # Сохраняем в app.context или используем globals, если не хочется передавать аргументы в каждый blueprint
    app.model_manager = model_manager
    app.data_adapter = data_adapter
    app.visualizer = visualizer
    app.feature_info = feature_info
    app.pipeline = pipe
    app.REQUIRED_FEATURES = feature_info['numeric_features'] + feature_info['categorical_features']

    # Регистрация blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(dataset_analys_bp, url_prefix='/dataset')

    # Создание папки uploads
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    return app

app = create_app()

if __name__ == '__main__':
    #app = create_app()
    #print(app.url_map)f
    app.run(debug=True, host='0.0.0.0', port=5000)
