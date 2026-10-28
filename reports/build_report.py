"""
Сборка отчёта: reports/report.md -> reports/report.pdf (PrinceXML)
===================================================================
1. Обновляет листинг кода в report.md (между маркерами LISTINGS) из lab1/.
2. Рендерит Markdown в HTML (mistune + подсветка pygments), строит оглавление.
3. Вызывает Prince для HTML -> PDF.

Запуск:
    python reports/build_report.py

Prince ищется в PATH, затем в ~/.local/bin; путь можно задать переменной PRINCE.
Диаграммы (reports/diagrams/*.svg) рендерятся отдельно из *.puml через PlantUML.
"""

import html
import os
import re
import shutil
import subprocess
import sys

import mistune
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(REPORTS_DIR)
MD_PATH = os.path.join(REPORTS_DIR, 'report.md')
PDF_PATH = os.path.join(REPORTS_DIR, 'report.pdf')

LISTINGS = [
    ('lab1/models/quadrotor_2d.xml', 'xml'),
    ('lab1/src/perception.py', 'python'),
    ('lab1/src/planning.py', 'python'),
    ('lab1/src/control.py', 'python'),
    ('lab1/src/fail_operational.py', 'python'),
]
LISTINGS_START = '<!-- LISTINGS:START -->'
LISTINGS_END = '<!-- LISTINGS:END -->'

CSS = """
@page {
  size: A4;
  margin: 20mm 15mm 20mm 30mm;
  @bottom-center { content: counter(page); font: 10pt "Liberation Serif", serif; }
}
@page:first { @bottom-center { content: none; } }

body {
  font: 12pt/1.38 "Liberation Serif", "Times New Roman", serif;
  text-align: justify;
  hyphens: auto;
  color: #000;
}
p { margin: 0 0 6pt; text-indent: 0; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }

h2 {
  font-size: 14pt; margin: 16pt 0 8pt;
  page-break-after: avoid;
  prince-bookmark-level: 1;
}
h3 {
  font-size: 12pt; margin: 12pt 0 6pt;
  page-break-after: avoid;
  prince-bookmark-level: 2;
}

table {
  border-collapse: collapse; width: 100%;
  margin: 0 0 10pt; font-size: 10.5pt; line-height: 1.25;
  page-break-inside: avoid; text-align: left;
}
th, td { border: 0.6pt solid #555; padding: 3pt 5pt; vertical-align: top; }
th { background: #e8e8e8; }
.tcap { margin: 8pt 0 3pt; page-break-after: avoid; }

figure { margin: 8pt 0 10pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 11pt; margin-top: 4pt; text-align: center; }

code { font-family: "DejaVu Sans Mono", monospace; font-size: 9.5pt; }
pre {
  font: 9pt/1.3 "DejaVu Sans Mono", monospace;
  background: #f6f6f6; border: 0.5pt solid #ccc;
  padding: 5pt 7pt; margin: 0 0 8pt;
  white-space: pre-wrap; text-align: left;
}
.highlight pre { page-break-inside: avoid; }
.listings ~ .highlight pre,
.listings ~ pre {
  font-size: 7.2pt; line-height: 1.22;
  page-break-inside: auto;
}

/* Титульный лист */
.title-page {
  page-break-after: always;
  height: 252mm; position: relative; text-align: center;
}
.tp-top p { font-size: 13pt; margin: 0; }
.tp-center { margin-top: 62mm; }
.tp-kind { font-size: 18pt; font-weight: bold; line-height: 1.4; margin-bottom: 14pt; }
.tp-name { font-size: 15pt; font-weight: bold; line-height: 1.4; margin: 0 8mm 14pt; text-align: center; }
.tp-variant { font-size: 13pt; line-height: 1.4; }
.tp-right { margin: 40mm 0 0 82mm; text-align: left; font-size: 12.5pt; }
.tp-right p { margin: 0 0 5pt; }
.tp-bottom { position: absolute; bottom: 0; width: 100%; font-size: 13pt; }
.tp-repo { font-size: 11pt; margin-bottom: 10pt; }
.tp-repo a { color: #1a4f9c; }

/* Оглавление */
.toc { page-break-after: always; }
.toc h2 { text-align: center; prince-bookmark-level: none; }
.toc ul { list-style: none; padding: 0; }
.toc li { margin: 0 0 5pt; }
.toc a { color: #000; text-decoration: none; }
.toc a::after { content: leader('.') target-counter(attr(href), page); }

.page-break { page-break-before: always; }
a { color: #000; text-decoration: none; }
""" + HtmlFormatter(style='default').get_style_defs('.highlight')


def update_listings(md):
    """Вставить актуальный код из lab1/ между маркерами LISTINGS."""
    parts = ['<div class="listings"></div>', '']
    for i, (path, lang) in enumerate(LISTINGS, 1):
        with open(os.path.join(ROOT_DIR, path), encoding='utf-8') as f:
            code = f.read().rstrip('\n')
        parts += [f'### А.{i}. `{path}`', '', f'```{lang}', code, '```', '']
    block = LISTINGS_START + '\n' + '\n'.join(parts) + LISTINGS_END
    pattern = re.escape(LISTINGS_START) + r'.*?' + re.escape(LISTINGS_END)
    if not re.search(pattern, md, flags=re.S):
        sys.exit('В report.md нет маркеров LISTINGS:START/END')
    return re.sub(pattern, lambda _: block, md, flags=re.S)


class ReportRenderer(mistune.HTMLRenderer):
    """HTML-рендерер: подсветка кода и якоря у заголовков для оглавления."""

    def __init__(self):
        super().__init__(escape=False)
        self.headings = []

    def heading(self, text, level, **attrs):
        if level != 2:
            return f'<h{level}>{text}</h{level}>\n'
        anchor = f'sec-{len(self.headings) + 1}'
        self.headings.append((anchor, text))
        return f'<h2 id="{anchor}">{text}</h2>\n'

    def block_code(self, code, info=None):
        lang = (info or 'text').split()[0]
        try:
            lexer = get_lexer_by_name(lang)
        except ClassNotFound:
            return f'<pre>{html.escape(code)}</pre>\n'
        return highlight(code, lexer, HtmlFormatter(nowrap=False))


def render_html(md):
    renderer = ReportRenderer()
    body = mistune.create_markdown(renderer=renderer, plugins=['table'])(md)
    toc_items = '\n'.join(f'<li><a href="#{a}">{re.sub(r"<[^>]+>", "", t)}</a></li>'
                          for a, t in renderer.headings)
    toc = f'<div class="toc"><h2>Содержание</h2><ul>{toc_items}</ul></div>'
    body = body.replace('<!-- TOC -->', toc)
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<title>Отчёт по лабораторной работе №1</title>'
            f'<meta name="author" content="Цароев Альберт Казебкович">'
            f'<style>{CSS}</style></head><body>{body}</body></html>')


def find_prince():
    candidates = [os.environ.get('PRINCE'), shutil.which('prince'),
                  os.path.expanduser('~/.local/bin/prince')]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    sys.exit('Prince не найден: установите PrinceXML или задайте переменную PRINCE')


def main():
    with open(MD_PATH, encoding='utf-8') as f:
        md = f.read()
    md = update_listings(md)
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md)

    page = render_html(md)
    result = subprocess.run(
        [find_prince(), '-', '-o', PDF_PATH, f'--baseurl={REPORTS_DIR}/'],
        input=page.encode('utf-8'), capture_output=True)
    sys.stderr.write(result.stderr.decode('utf-8', 'replace'))
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f'Готово: {os.path.relpath(PDF_PATH, ROOT_DIR)}')


if __name__ == '__main__':
    main()
