# modules_site/models/__init__.py
from flask import Blueprint

# Создаем Blueprint с префиксом URL '/models'
bp = Blueprint('models', __name__, url_prefix='/models', template_folder='../../templates')

# Импортируем роуты после создания bp
from . import routes