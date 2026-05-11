from flask import Blueprint

bp = Blueprint('dataset_analys', __name__, template_folder='../../templates')
from . import routes