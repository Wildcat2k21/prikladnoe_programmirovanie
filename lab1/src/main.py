"""
Лабораторная работа №1 — Вариант №10 (2D-квадрокоптер, висение в точке, PD по высоте)
=======================================================================================
Полный замкнутый контур: Perception -> [Fail-Operational] -> Planning -> Control

Эксперименты:
    1. Baseline: полёт без контроллера
    2. Perception: истина vs зашумлённый vs задержанный сигнал
    3. Planning: жёсткая траектория по waypoints
    4. Частота управления -> устойчивость
    5. Задержка измерения -> устойчивость
    6. Fail-Operational: отказ высотомера

Запуск (из любой папки):
    python lab1/src/main.py          # сохранить графики и таблицы в reports/
    python lab1/src/main.py --show   # дополнительно показать окна графиков
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

# Скрипт без --show рисует в файлы; при импорте из ноутбука backend не трогаем
if __name__ == '__main__' and '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from perception import PerceptionLayer
from planning import HoverPDPlanner, WaypointPlanner, quadrotor_params
from control import (ControlLoop, run_episode, reset,
                     control_freq_experiment, control_delay_experiment,
                     START, TARGET)
from fail_operational import FailOperationalEstimator

ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..', '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'lab1', 'models', 'quadrotor_2d.xml')
FIG_DIR = os.path.join(ROOT_DIR, 'reports', 'figures')
RESULTS_PATH = os.path.join(ROOT_DIR, 'reports', 'results.md')

SIM_TIME = 10.0
FREQUENCIES = [1000, 500, 200, 100, 50, 20]
DELAYS_MS = [0, 5, 10, 20, 50, 100, 200]
FAULT_TIME = 3.0
FAULT_SIGMA = 0.1
NOMINAL_SIGMA = 0.002

FREQ_COMMENTS = {1000: 'Как Atlas', 500: 'Как Unitree', 200: 'Веб-«нормально»'}


def load_model():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    return model, mujoco.MjData(model)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    print(f"  график: {os.path.relpath(path, ROOT_DIR)}")


def fmt_time(log):
    return f"{log['fall_time']:.2f} с" if not log['stable'] else f"> {SIM_TIME:.0f} с (не упал)"


def transient(log):
    return (f"θ установился за {log['theta_settle']:.2f} с, "
            f"перерегулирование {log['theta_overshoot']:.3f} рад")


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def print_table(header, rows):
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print('  ' + ' | '.join(str(c).ljust(w) for c, w in zip(row, widths)))


def experiment_baseline(model, data):
    """Эксперимент 1: без контроллера (моторы выключены / постоянная тяга висения)."""
    print("\n=== ЭКСПЕРИМЕНТ 1: полёт без контроллера ===")
    mass, _, _ = quadrotor_params(model)
    hover = mass * 9.81 / 2
    off = run_episode(model, data, open_loop_forces=[0.0, 0.0], sim_time=SIM_TIME)
    open_loop = run_episode(model, data, open_loop_forces=[hover, hover], sim_time=SIM_TIME)

    print(f"  шаг физики: {model.opt.timestep * 1000:.1f} мс ({1 / model.opt.timestep:.0f} Гц)")
    print(f"  моторы выключены:           падение через {off['fall_time']:.3f} с ({off['reason']})")
    print(f"  постоянная тяга {hover:.2f} Н x2: падение через {open_loop['fall_time']:.3f} с ({open_loop['reason']})")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(off['t'], off['z'], 'r-', label=f"Моторы выключены (падение {off['fall_time']:.2f} с)")
    axes[0].plot(open_loop['t'], open_loop['z'], 'b-',
                 label=f"Тяга висения без обратной связи (падение {open_loop['fall_time']:.2f} с)")
    axes[0].axhline(TARGET[1], color='g', ls=':', label='Целевая высота')
    axes[0].axhline(0.05, color='k', ls='--', alpha=0.5, label='Земля')
    axes[0].set_ylabel('Высота z, м')
    axes[0].set_title('Эксперимент 1. Полёт без контроллера (baseline), старт z=1 м, θ=0.1 рад')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(off['t'], off['x'], 'r-', label='Моторы выключены')
    axes[1].plot(open_loop['t'], open_loop['x'], 'b-', label='Тяга висения без обратной связи')
    axes[1].set_xlabel('Время, с'); axes[1].set_ylabel('Положение x, м')
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    save(fig, 'exp1_baseline.png')
    return {'off': off, 'open_loop': open_loop, 'hover_force': hover}


def experiment_perception(model, data):
    """Эксперимент 2 (п. 2.4): истина vs зашумлённый vs задержанный; камера 30 fps vs IMU."""
    print("\n=== ЭКСПЕРИМЕНТ 2: слой Perception ===")
    sim_dt = model.opt.timestep
    delay_ms, sigma = 20, 0.02
    planner = HoverPDPlanner(*quadrotor_params(model))
    loop = ControlLoop(model, 500)
    control_sensors = PerceptionLayer(sim_dt)  # управляем по идеальным сенсорам
    demo = PerceptionLayer(sim_dt, delay_steps=round(delay_ms / 1000 / sim_dt),
                           sigma_imu=sigma, sigma_alt=sigma, sigma_pos=sigma,
                           pos_rate_hz=30, seed=1)
    reset(model, data)
    log = {k: [] for k in ('t', 'z_true', 'z_noisy', 'z_delayed',
                           'x_true', 'x_cam', 'th_true', 'th_imu')}
    for step in range(int(2.0 / sim_dt)):
        obs = control_sensors.observe(data)
        sensed = demo.observe(data)
        if loop.is_control_step(step):
            forces, _ = planner.compute(obs, TARGET)
        loop.apply(data, forces)
        loop.step_physics(data)
        log['t'].append(step * sim_dt)
        log['z_true'].append(sensed['z_true']); log['z_noisy'].append(sensed['z_raw'])
        log['z_delayed'].append(sensed['z'])
        log['x_true'].append(sensed['x_true']); log['x_cam'].append(sensed['x'])
        log['th_true'].append(sensed['theta_true']); log['th_imu'].append(sensed['theta'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(log['t'], log['z_noisy'], color='0.7', lw=0.6, label=f'Зашумлённый (σ={sigma} м)')
    axes[0].plot(log['t'], log['z_delayed'], 'b-', lw=0.8, label=f'Зашумлённый + задержка {delay_ms} мс')
    axes[0].plot(log['t'], log['z_true'], 'k-', lw=1.5, label='Истинная высота')
    axes[0].set_xlim(0.1, 0.5); axes[0].set_ylim(1.0, 1.45)
    axes[0].set_xlabel('Время, с'); axes[0].set_ylabel('Высота z, м')
    axes[0].set_title('Высотомер: истина vs зашумлённый vs задержанный (фрагмент набора высоты)')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(log['t'], log['th_imu'], color='0.6', lw=0.6, label='IMU 1000 Гц (θ, рад)')
    axes[1].plot(log['t'], log['th_true'], 'k-', lw=1.2, label='Истинный угол θ, рад')
    axes[1].step(log['t'], log['x_cam'], 'r-', lw=1.0, where='post', label='Камера 30 fps (x, м)')
    axes[1].plot(log['t'], log['x_true'], 'r:', lw=1.2, label='Истинное x, м')
    axes[1].set_xlabel('Время, с'); axes[1].set_ylabel('θ, рад / x, м')
    axes[1].set_title('Рассинхронизация потоков: камера 30 fps vs IMU 1000 Гц (+ задержка 20 мс)')
    axes[1].legend(loc='upper right'); axes[1].grid(True)
    fig.tight_layout()
    save(fig, 'exp2_perception.png')
    return {'delay_ms': delay_ms, 'sigma': sigma}


def experiment_waypoints(model, data):
    """Эксперимент 3 (п. 3.3): жёсткая траектория по waypoints."""
    print("\n=== ЭКСПЕРИМЕНТ 3: жёсткая траектория по waypoints ===")
    points = [(0.0, 1.5), (0.5, 1.5), (0.5, 1.0), (0.0, 1.0), (0.0, 1.5)]
    log = run_episode(model, data, waypoints=WaypointPlanner(points, tolerance=0.05),
                      sim_time=15.0)
    print(f"  маршрут из {len(points)} точек, результат: {'пройден' if log['stable'] else log['reason']}")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(log['x'], log['z'], 'b-', label='Фактическая траектория')
    ax.plot(*zip(*points), 'ro--', label='Waypoints (заданы до старта)')
    ax.plot(START['x'], START['z'], 'ks', label='Старт')
    ax.set_xlabel('x, м'); ax.set_ylabel('z, м'); ax.axis('equal')
    ax.set_title('Эксперимент 3. Полёт по жёсткой траектории (waypoints)')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    save(fig, 'exp3_waypoints.png')
    return log


def plot_family(logs, label_fmt, title, name):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, log in logs.items():
        label = label_fmt(key) + (f" (падение: {log['fall_time']:.2f} с)" if not log['stable'] else ' (устойчиво)')
        axes[0].plot(log['t'], log['theta'], lw=1, label=label)
        axes[1].plot(log['t'], log['z'], lw=1, label=label)
    axes[0].axhline(np.pi / 2, color='k', ls='--', alpha=0.3)
    axes[0].axhline(-np.pi / 2, color='k', ls='--', alpha=0.3)
    axes[0].set_ylabel('Угол θ, рад'); axes[0].set_title(title)
    axes[0].set_ylim(-2, 2); axes[0].legend(fontsize=8); axes[0].grid(True)
    axes[1].axhline(TARGET[1], color='g', ls=':')
    axes[1].set_xlabel('Время, с'); axes[1].set_ylabel('Высота z, м')
    axes[1].legend(fontsize=8); axes[1].grid(True)
    fig.tight_layout()
    save(fig, name)


def delay_margin(model, data, freq_hz, max_ms=30):
    """Максимальная дополнительная задержка измерения (мс), которую выдерживает контур."""
    margin = 0
    for d in range(1, max_ms + 1):
        if not control_delay_experiment(model, data, d, freq_hz, SIM_TIME)['stable']:
            break
        margin = d
    return margin


def experiment_frequency(model, data):
    """Эксперимент 4 (п. 4.2-4.3): частота управления -> устойчивость."""
    print("\n=== ЭКСПЕРИМЕНТ 4: частота управления ===")
    logs = {f: control_freq_experiment(model, data, f, SIM_TIME) for f in FREQUENCIES}

    # Граница: перебор периода управления с шагом 1 мс (частота кратна шагу физики)
    stable_periods = [p for p in range(20, 51)
                      if control_freq_experiment(model, data, 1000 / p, SIM_TIME)['stable']]
    worst_ok = max(stable_periods)
    critical = {'min_stable_hz': 1000 / worst_ok, 'max_unstable_hz': 1000 / (worst_ok + 1)}

    rows = []
    for f, log in logs.items():
        comment = FREQ_COMMENTS.get(f, '')
        if log['stable']:
            comment += (' — ' if comment else '') + \
                f"запас по задержке {delay_margin(model, data, f)} мс; " + transient(log)
        else:
            comment += (' — ' if comment else '') + log['reason']
        rows.append([f'{f} Гц', fmt_time(log), 'Да' if log['stable'] else 'Нет', comment])
    print_table(['Частота', 'Время до падения', 'Устойчиво?', 'Комментарий'], rows)
    print(f"  критическая частота: устойчиво при >= {critical['min_stable_hz']:.1f} Гц, "
          f"падение при <= {critical['max_unstable_hz']:.1f} Гц")
    plot_family(logs, lambda f: f'{f} Гц',
                'Эксперимент 4. Влияние частоты управления на устойчивость', 'exp4_frequency.png')
    return logs, rows, critical


def experiment_delay(model, data):
    """Эксперимент 5 (п. 4.4): задержка измерения при 500 Гц -> устойчивость."""
    print("\n=== ЭКСПЕРИМЕНТ 5: задержка (500 Гц) ===")
    logs = {d: control_delay_experiment(model, data, d, 500, SIM_TIME) for d in DELAYS_MS}
    stable_delays = [d for d in range(0, 21)
                     if control_delay_experiment(model, data, d, 500, SIM_TIME)['stable']]
    critical = {'max_stable_ms': max(stable_delays), 'min_unstable_ms': max(stable_delays) + 1}

    rows = []
    for d, log in logs.items():
        if log['stable']:
            comment = transient(log)
        else:
            comment = log['reason']
        if d == 20:
            comment += ' — «20 мс у Atlas = падение с высоты»'
        rows.append([f'{d} мс', fmt_time(log), 'Да' if log['stable'] else 'Нет', comment])
    print_table(['Задержка', 'Время до падения', 'Устойчиво?', 'Комментарий'], rows)
    print(f"  граница устойчивости: {critical['max_stable_ms']} мс — ещё летит, "
          f"{critical['min_unstable_ms']} мс — падает")
    plot_family(logs, lambda d: f'{d} мс',
                'Эксперимент 5. Влияние задержки измерения на устойчивость (500 Гц)',
                'exp5_delay.png')
    return logs, rows, critical


def fault_episode(model, data, sigma, detector, mass_scale=1.0):
    sim_dt = model.opt.timestep
    mass, _, _ = quadrotor_params(model)
    perception = PerceptionLayer(sim_dt, sigma_imu=NOMINAL_SIGMA, sigma_alt=NOMINAL_SIGMA,
                                 sigma_pos=NOMINAL_SIGMA)
    perception.inject_altimeter_fault(FAULT_TIME, sigma)
    estimator = FailOperationalEstimator(sim_dt, mass, model_mass_scale=mass_scale) if detector else None
    log = run_episode(model, data, 500, perception=perception, estimator=estimator,
                      sim_time=SIM_TIME)
    log['detected_at'] = log['t'][log['fault_flag']][0] if log['fault_flag'].any() else None
    return log


def experiment_fail_operational(model, data):
    """Эксперимент 6 (часть 5): отказ высотомера в t=3 с."""
    print("\n=== ЭКСПЕРИМЕНТ 6: Fail-Operational (отказ высотомера) ===")
    no_det = fault_episode(model, data, FAULT_SIGMA, detector=False)
    det = fault_episode(model, data, FAULT_SIGMA, detector=True)
    det_err = fault_episode(model, data, FAULT_SIGMA, detector=True, mass_scale=1.02)

    def comment(log, base):
        tail = f"ошибка висения {log['hover_error'] * 100:.1f} см" if log['stable'] else log['reason']
        detect = (f", отказ обнаружен через {(log['detected_at'] - FAULT_TIME) * 1000:.0f} мс"
                  if log['detected_at'] is not None else '')
        return f"{base}: {tail}{detect}"

    rows = [
        ['Без детектора отказа', fmt_time(no_det), comment(no_det, 'Fail-Safe: система падает')],
        ['С детектором отказа', fmt_time(det),
         comment(det, 'Fail-Operational: система продолжает работу')],
        ['С детектором, ошибка массы в модели 2%', fmt_time(det_err),
         comment(det_err, 'Деградация: резерв по модели дрейфует')],
    ]
    print(f"  отказ высотомера в t={FAULT_TIME} с, шум σ={FAULT_SIGMA} м (номинал σ={NOMINAL_SIGMA} м)")
    print_table(['Режим', f'Время до падения при σ={FAULT_SIGMA}', 'Комментарий'], rows)

    sweep_rows = []
    for sigma in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        a = fault_episode(model, data, sigma, detector=False)
        b = fault_episode(model, data, sigma, detector=True)
        sweep_rows.append([f'{sigma}', fmt_time(a),
                           f"{a['hover_error'] * 100:.1f} см" if a['stable'] else a['reason'],
                           fmt_time(b),
                           f"{b['hover_error'] * 100:.1f} см" if b['stable'] else b['reason']])
    print("  зависимость от силы отказа:")
    print_table(['σ отказа, м', 'Без детектора', 'Итог', 'С детектором', 'Итог'], sweep_rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, log, title in [(axes[0], no_det, 'Без детектора отказа (Fail-Safe)'),
                           (axes[1], det, 'С детектором отказа (Fail-Operational)')]:
        ax.plot(log['t'], log['z_meas'], color='0.75', lw=0.5, label='Показания высотомера')
        if log is det:
            ax.plot(log['t'], log['z_used'], 'b-', lw=1, label='Высота, используемая контроллером')
        ax.plot(log['t'], log['z'], 'k-', lw=1.5, label='Истинная высота')
        ax.axvline(FAULT_TIME, color='r', ls='--', label=f'Отказ высотомера (σ={FAULT_SIGMA} м)')
        if log['detected_at'] is not None:
            ax.axvline(log['detected_at'], color='orange', ls=':',
                       label=f"Отказ обнаружен ({log['detected_at']:.3f} с)")
        if not log['stable']:
            ax.axvline(log['fall_time'], color='k', ls='-.', label=f"Потеря устойчивости ({log['fall_time']:.2f} с)")
        ax.set_ylim(0, 3); ax.set_ylabel('Высота z, м'); ax.set_title(title)
        ax.legend(fontsize=8, loc='upper left'); ax.grid(True)
    axes[1].set_xlabel('Время, с')
    fig.tight_layout()
    save(fig, 'exp6_fail_operational.png')
    return rows, sweep_rows


def write_results(baseline, freq, delay, fail_op, model):
    mass, inertia, arm = quadrotor_params(model)
    _, freq_rows, freq_crit = freq
    _, delay_rows, delay_crit = delay
    fo_rows, fo_sweep = fail_op
    text = f"""# Результаты экспериментов — вариант №10

> Файл сгенерирован `python lab1/src/main.py`. Не редактируйте вручную — перезапустите скрипт.

**Объект:** 2D-квадрокоптер, масса {mass:.3f} кг, I_yy = {inertia:.4f} кг·м², плечо {arm} м.
**Задача:** висение в точке (x, z) = {TARGET} м, старт из (x, z, θ) = ({START['x']}, {START['z']}, {START['theta']} рад).
**Шаг физики:** {model.opt.timestep * 1000:.1f} мс ({1 / model.opt.timestep:.0f} Гц).
**Критерии падения:** z < 0.05 м (удар о землю), |θ| > π/2 (переворот), отклонение от точки > 1 м (улёт).

## Эксперимент 1. Без контроллера

| Режим | Время до падения | Причина |
|---|---|---|
| Моторы выключены | {baseline['off']['fall_time']:.3f} с | {baseline['off']['reason']} |
| Постоянная тяга висения {baseline['hover_force']:.2f} Н на мотор (без обратной связи) | {baseline['open_loop']['fall_time']:.3f} с | {baseline['open_loop']['reason']} |

![](figures/exp1_baseline.png)

## Эксперимент 2. Perception

![](figures/exp2_perception.png)

## Эксперимент 3. Жёсткая траектория (waypoints)

![](figures/exp3_waypoints.png)

## Эксперимент 4. Частота управления → устойчивость

{md_table(['Частота управления', 'Время до падения', 'Устойчиво?', 'Комментарий'], freq_rows)}

**Критическая частота:** устойчиво при ≥ {freq_crit['min_stable_hz']:.1f} Гц, падение при ≤ {freq_crit['max_unstable_hz']:.1f} Гц.

![](figures/exp4_frequency.png)

## Эксперимент 5. Задержка → устойчивость (500 Гц)

{md_table(['Задержка', 'Время до падения', 'Устойчиво?', 'Комментарий'], delay_rows)}

**Граница устойчивости:** {delay_crit['max_stable_ms']} мс — ещё летит, {delay_crit['min_unstable_ms']} мс — падает.

![](figures/exp5_delay.png)

## Эксперимент 6. Fail-Operational (отказ высотомера в t = {FAULT_TIME} с)

{md_table(['Режим', f'Время до падения при шуме σ={FAULT_SIGMA}', 'Комментарий'], fo_rows)}

Зависимость от силы отказа:

{md_table(['σ отказа, м', 'Без детектора', 'Итог', 'С детектором', 'Итог'], fo_sweep)}

![](figures/exp6_fail_operational.png)
"""
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nТаблицы: {os.path.relpath(RESULTS_PATH, ROOT_DIR)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--show', action='store_true', help='показать окна графиков')
    args = parser.parse_args()

    print("Лабораторная работа №1 — вариант №10: 2D-квадрокоптер, висение в точке")
    print(f"MuJoCo {mujoco.__version__}, модель: {os.path.relpath(MODEL_PATH, ROOT_DIR)}")
    model, data = load_model()

    baseline = experiment_baseline(model, data)
    experiment_perception(model, data)
    experiment_waypoints(model, data)
    freq = experiment_frequency(model, data)
    delay = experiment_delay(model, data)
    fail_op = experiment_fail_operational(model, data)
    write_results(baseline, freq, delay, fail_op, model)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
