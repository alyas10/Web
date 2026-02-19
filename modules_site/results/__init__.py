# modules_site/results/__init__.py
from flask import Blueprint

# Создаем Blueprint с префиксом URL '/results'
bp = Blueprint('results', __name__, url_prefix='/results', template_folder='../../templates')

# Импортируем роуты после создания bp
from . import routes