import os
import json
from flask import render_template, request, jsonify, current_app
from . import bp

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'app_config.json')

def load_config():
    """Загружает настройки из JSON файла"""
    if not os.path.exists(CONFIG_FILE):
        # Возвращаем дефолтные значения, если файла нет
        return {
            "project_name": "ML Network Security Project",
            "project_description": "",
            "max_file_size_mb": 500,
            "auto_preprocess": True,
            "normalize_features": True,
            "balance_classes": True,
            "notify_complete": True,
            "notify_threats": True,
            "notify_errors": False
        }
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(data):
    """Сохраняет настройки в JSON файл"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


@bp.route('/', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Получаем данные из формы
        data = request.form

        # Формируем словарь настроек (преобразуем типы данных)
        new_config = {
            "project_name": data.get('project_name'),
            "project_description": data.get('project_description'),
            "max_file_size_mb": int(data.get('max_file_size_mb', 500)),
            "auto_preprocess": data.get('auto_preprocess') == 'on',
            "normalize_features": data.get('normalize_features') == 'on',
            "balance_classes": data.get('balance_classes') == 'on',
            "notify_complete": data.get('notify_complete') == 'on',
            "notify_threats": data.get('notify_threats') == 'on',
            "notify_errors": data.get('notify_errors') == 'on'
        }

        try:
            save_config(new_config)
            # Можно вернуть JSON ответ для AJAX или просто перенаправить
            return jsonify({"status": "success", "message": "Настройки успешно сохранены!"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

        # Если метод GET - загружаем текущие настройки и передаем их в шаблон
    config = load_config()
    return render_template("settings.html", config=config)