# modules_site/pipeline/__init__.py
from flask import Blueprint

# Создаем Blueprint с префиксом URL '/pipeline'
bp = Blueprint('models', __name__, url_prefix='/models', template_folder='../../templates')

# Импортируем роуты после создания bp
from . import routes