import os
import json
from flask import render_template, request, jsonify, current_app
from . import bp

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'app_config.json')

DEFAULT_CONFIG = {
    "project_name": "ML Network Security Project",
    "project_description": "",
    "max_file_size_mb": 1000,
    "auto_preprocess": True,
    "normalize_features": True,
    "balance_classes": True,
    "train_test_split": 80,
    "notify_complete": True,
    "notify_threats": True,
    "notify_errors": False,
    "theme": "dark"
}


def load_config():
    """Загружает настройки из JSON файла"""
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Объединяем с дефолтными значениями на случай добавления новых настроек
            return {**DEFAULT_CONFIG, **config}
    except (json.JSONDecodeError, IOError) as e:
        current_app.logger.error(f"Ошибка загрузки конфига: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(data):
    """Сохраняет настройки в JSON файл с валидацией"""
    # Валидация и преобразование типов
    validated_config = {
        "project_name": str(data.get('project_name', DEFAULT_CONFIG['project_name']))[:100],
        "project_description": str(data.get('project_description', ''))[:500],
        "max_file_size_mb": max(10, min(1000, int(data.get('max_file_size_mb', DEFAULT_CONFIG['max_file_size_mb'])))),
        "auto_preprocess": data.get('auto_preprocess') == 'on',
        "normalize_features": data.get('normalize_features') == 'on',
        "balance_classes": data.get('balance_classes') == 'on',
        "train_test_split": max(50, min(90, int(data.get('train_test_split', DEFAULT_CONFIG['train_test_split'])))),
        "notify_complete": data.get('notify_complete') == 'on',
        "notify_threats": data.get('notify_threats') == 'on',
        "notify_errors": data.get('notify_errors') == 'on',
        "theme": data.get('theme', DEFAULT_CONFIG['theme'])
    }

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(validated_config, f, indent=4, ensure_ascii=False)
    return validated_config


@bp.route('/', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Получаем данные из формы
        data = request.form

        try:
            new_config = save_config(data)
            # Обновляем MAX_CONTENT_LENGTH в приложении
            current_app.config['MAX_CONTENT_LENGTH'] = new_config['max_file_size_mb'] * 1024 * 1024

            return jsonify({
                "status": "success",
                "message": "Настройки успешно сохранены!",
                "config": new_config
            })
        except ValueError as e:
            return jsonify({"status": "error", "message": f"Ошибка валидации: {str(e)}"}), 400
        except Exception as e:
            current_app.logger.error(f"Ошибка сохранения настроек: {e}")
            return jsonify({"status": "error", "message": "Внутренняя ошибка сервера"}), 500

    # Если метод GET - загружаем текущие настройки и передаем их в шаблон
    config = load_config()
    return render_template("settings.html", config=config)


@bp.route('/reset', methods=['POST'])
def reset_settings():
    """Сбрасывает настройки к значениям по умолчанию"""
    try:
        save_config(DEFAULT_CONFIG)
        return jsonify({
            "status": "success",
            "message": "Настройки сброшены к значениям по умолчанию!"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route('/api/config', methods=['GET'])
def get_config_api():
    """API endpoint для получения текущих настроек"""
    config = load_config()
    return jsonify({"status": "success", "config": config})
