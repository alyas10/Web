# modules_site/dashboard/__init__.py
from flask import Blueprint

# Создаем Blueprint с префиксом URL '/'
# Указываем папку с шаблонами относительно текущего пакета
bp = Blueprint('dashboard', __name__, template_folder='../../templates')

# Импортируем роуты после создания blueprint
from . import routes