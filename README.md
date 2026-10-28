# Лабораторная работа №1
## «Первый контакт с физикой: настройка симуляции, слои Perception → Planning → Control и физический дедлайн»

**Вариант №10:** 2D-квадрокоптер — висение в точке, PD по высоте.

Архитектура: **Perception → [Fail-Operational] → Planning → Control** на MuJoCo 3.x.

---

## Окружение

| Компонент | Версия (проверено) |
|---|---|
| Python | 3.10+ (проверено на 3.14.4) |
| MuJoCo | 3.x (проверено на 3.17.0) |
| NumPy | проверено на 2.5.4 |
| Matplotlib | проверено на 3.11.2 |
| Jupyter Notebook | для `lab1/notebooks/` |
| IDE | VS Code + расширения Python, Pylance, Jupyter (или PyCharm) |

### Установка

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import mujoco; print(mujoco.__version__)"
```

Если `python3 -m venv .venv` падает с ошибкой про `ensurepip` (Ubuntu/Debian без пакета `python3-venv`):

```bash
python3 -m venv --without-pip .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
```

В VS Code: `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv`.

---

## Запуск

```bash
python lab1/src/main.py          # все эксперименты → reports/figures/*.png и reports/results.md
python lab1/src/main.py --show   # то же + окна с графиками
```

Ноутбук: `lab1/notebooks/lab1_experiments.ipynb` (вызывает те же функции, что и `main.py`).

---

## Структура

```
├── askme.md                     # Контрольные вопросы и ответы (по результатам экспериментов)
├── requirements.txt
├── lab1/
│   ├── models/quadrotor_2d.xml  # 2D-квадрокоптер: x, z, pitch; 2 мотора; шаг физики 1 мс
│   ├── src/
│   │   ├── perception.py        # Шум, задержка, камера 30 fps vs IMU, высотомер + производная
│   │   ├── planning.py          # Каскадный PD: высота → тяга, x → угол, угол → момент; waypoints
│   │   ├── control.py           # ControlLoop с фиксированной частотой, эпизод, эксперименты 4.2/4.4
│   │   ├── fail_operational.py  # Детектор отказа по дисперсии + предсказание высоты по модели
│   │   └── main.py              # Все эксперименты, графики, таблицы
│   └── notebooks/lab1_experiments.ipynb
└── reports/
    ├── results.md               # Сгенерированные таблицы (все ячейки заполнены)
    └── figures/                 # Графики для отчёта
```

---

## Эксперименты

| № | Что | Пункт методички | Файл |
|---|---|---|---|
| 1 | Полёт без контроллера (baseline) | 1.3–1.4 | `exp1_baseline.png` |
| 2 | Истина vs зашумлённый vs задержанный; камера 30 fps vs IMU 1000 Гц | 2.4 | `exp2_perception.png` |
| 3 | Жёсткая траектория по waypoints | 3.3 | `exp3_waypoints.png` |
| 4 | Частота управления → устойчивость (1000…20 Гц) + критическая частота | 4.2–4.3 | `exp4_frequency.png` |
| 5 | Задержка 0…200 мс при 500 Гц + граница устойчивости | 4.4 | `exp5_delay.png` |
| 6 | Fail-Safe vs Fail-Operational при отказе высотомера (σ = 0.1 м) | 5.1–5.3 | `exp6_fail_operational.png` |

Ключевые результаты (подробно в `reports/results.md`):
- висение устойчиво при 50–1000 Гц, при 20 Гц аппарат переворачивается; критическая частота ≈ 23 Гц;
- при 500 Гц граница устойчивости по задержке — **17 мс**, при 20 мс аппарат переворачивается;
- при отказе высотомера без детектора точка висения теряется через 1.1 с, с детектором полёт
  продолжается с ошибкой 2 см.

---

## Отчёт

Готовый отчёт: [`reports/report.pdf`](reports/report.pdf) (исходник — [`reports/report.md`](reports/report.md)).

```bash
python lab1/src/main.py            # пересчитать графики и таблицы
python reports/build_report.py     # обновить листинг в report.md и собрать PDF (нужен PrinceXML)
```

Диаграммы — `reports/diagrams/*.puml` (PlantUML), отрендерены в `*.svg`.

Требования методички: формат `.pdf` или `.docx`, 8–15 страниц. Обязательные разделы: титульный лист (ФИО, группа, вариант №10, IDE),
цель и задачи, схема трёх слоёв, эксперименты 1–4 (графики и таблицы из `reports/`), ответы на 10 контрольных
вопросов (`askme.md`), рефлексия, листинг кода (`lab1/src/`).
