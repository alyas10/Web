import os
import re
import markdown
from flask import render_template, current_app, url_for
from . import bp


@bp.route('/')
def dataset_analys():
    report_path = os.path.join(current_app.root_path, 'docs', 'report.md')

    content = "<p>Файл <code>report.md</code> не найден в корне проекта.</p>"

    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Заменяем относительные пути к изображениям на URL Flask
        # Ищем все ![alt](path/to/image.png) и заменяем path на url_for
        def replace_image_path(match):
            alt_text = match.group(1)
            img_path = match.group(2)

            # Проверяем, что путь относительный (не начинается с http:// или /)
            if not img_path.startswith(('http://', 'https://', '/')):
                # Преобразуем в URL статического файла
                static_url = url_for('static', filename=img_path)
                return f'![{alt_text}]({static_url})'
            return match.group(0)

        # Regex для поиска markdown изображений
        md_content = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', replace_image_path, md_content)

        # Рендерим markdown
        content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'nl2br'])

    return render_template('dataset_analys.html', report_html=content)