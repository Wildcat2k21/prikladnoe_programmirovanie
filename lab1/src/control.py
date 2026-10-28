"""
Слой Control (Управление и исполнение) — 2D-квадрокоптер
==========================================================
Контур управления с фиксированной частотой и эксперименты
по частоте и задержке (часть 4 методички).

Демонстрирует:
- физический дедлайн: команда обновляется раз в 1/freq с, между
  обновлениями мотор держит старую команду (zero-order hold);
- необратимость ошибки: после «падения» эпизод можно только перезапустить.
"""

import mujoco
import numpy as np

from perception import PerceptionLayer
from planning import HoverPDPlanner, quadrotor_params

# Начальное состояние и цель висения (общие для всех экспериментов)
START = {'x': 0.0, 'z': 1.0, 'theta': 0.1}   # старт с наклоном 0.1 рад
TARGET = (0.0, 1.5)                          # точка висения (x, z), м

# Критерии потери устойчивости («падения»)
Z_GROUND = 0.05        # касание земли, м
THETA_FLIP = np.pi / 2 # переворот, рад
MAX_DEVIATION = 1.0    # улёт от точки висения, м


def reset(model, data, start=START):
    """Сброс эпизода — единственный способ «откатить» падение."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = [start['x'], start['z'], start['theta']]
    mujoco.mj_forward(model, data)


def failure_reason(data, target, check_deviation=True):
    """Причина потери устойчивости или None, если полёт продолжается."""
    x, z, theta = data.qpos
    if z < Z_GROUND:
        return 'удар о землю'
    if abs(theta) > THETA_FLIP:
        return 'переворот'
    if check_deviation and np.hypot(x - target[0], z - target[1]) > MAX_DEVIATION:
        return 'улёт от точки'
    return None


class ControlLoop:
    """
    Контур управления с заданной частотой.

    Физический дедлайн:
    - Atlas: 500-1000 Гц (1-2 мс)
    - Unitree: 500 Гц
    - «веб-нормально»: 200 Гц и ниже
    """

    def __init__(self, model, freq_hz=500):
        """
        Args:
            model: объект mujoco.MjModel
            freq_hz: частота обновления управления, Гц
        """
        self.model = model
        self.freq_hz = freq_hz
        self.sim_dt = model.opt.timestep
        # round, а не int: int(0.005 / 0.001) == 4 из-за погрешности float
        self.steps_per_control = max(1, round(1.0 / freq_hz / self.sim_dt))

    def is_control_step(self, step):
        return step % self.steps_per_control == 0

    def apply(self, data, forces):
        """Записать команду в исполнительные механизмы (держится до следующего обновления)."""
        data.ctrl[:] = forces

    def step_physics(self, data):
        mujoco.mj_step(self.model, data)


def run_episode(model, data, freq_hz=500, perception=None, planner=None,
                estimator=None, target=TARGET, sim_time=10.0,
                waypoints=None, open_loop_forces=None, start=START):
    """
    Полный замкнутый контур: Perception -> [Fail-Operational] -> Planning -> Control.

    Args:
        freq_hz: частота контура управления
        perception: PerceptionLayer (по умолчанию — идеальные сенсоры)
        planner: HoverPDPlanner (по умолчанию — штатные коэффициенты)
        estimator: FailOperationalEstimator или None (режим без детектора)
        waypoints: WaypointPlanner или None (висение в target)
        open_loop_forces: если задано — контроллер выключен, на моторы
            подаётся постоянная тяга (baseline, эксперимент 1)

    Returns:
        dict с логами и временем потери устойчивости fall_time
        (sim_time, если аппарат продержался весь эпизод)
    """
    reset(model, data, start)
    loop = ControlLoop(model, freq_hz)
    if perception is None:
        perception = PerceptionLayer(loop.sim_dt)
    if planner is None:
        planner = HoverPDPlanner(*quadrotor_params(model))

    log = {k: [] for k in ('t', 'x', 'z', 'theta', 'z_meas', 'z_used',
                           'f_left', 'f_right', 'fault_flag', 'target_x', 'target_z')}
    forces, thrust = np.zeros(2), 0.0
    current_target = target
    fall_time, reason = sim_time, None
    fault_flag = False

    for step in range(int(round(sim_time / loop.sim_dt))):
        t = step * loop.sim_dt

        # Perception: сенсоры работают на каждом шаге физики
        obs = perception.observe(data)
        used = obs
        if estimator is not None:
            used, fault_flag = estimator.update(obs, thrust)

        # Planning + Control: только в моменты обновления контура
        if open_loop_forces is not None:
            forces = np.asarray(open_loop_forces, dtype=float)
        elif loop.is_control_step(step):
            if waypoints is not None:
                current_target, _ = waypoints.get_target(used['x'], used['z'])
            forces, thrust = planner.compute(used, current_target)
        loop.apply(data, forces)
        loop.step_physics(data)

        log['t'].append(t)
        log['x'].append(data.qpos[0])
        log['z'].append(data.qpos[1])
        log['theta'].append(data.qpos[2])
        log['z_meas'].append(obs['z'])
        log['z_used'].append(used['z'])
        log['f_left'].append(forces[0])
        log['f_right'].append(forces[1])
        log['fault_flag'].append(fault_flag)
        log['target_x'].append(current_target[0])
        log['target_z'].append(current_target[1])

        # Без контроллера «улёт от точки» не считаем — ждём физического падения
        reason = failure_reason(data, target if waypoints is None else current_target,
                                check_deviation=open_loop_forces is None)
        if reason is not None:
            fall_time = t
            break

    log = {k: np.array(v) for k, v in log.items()}
    log['fall_time'] = fall_time
    log['reason'] = reason
    log['stable'] = reason is None
    # Ошибка висения за последнюю секунду (для устойчивых режимов)
    tail = log['t'] >= log['t'][-1] - 1.0
    log['hover_error'] = float(np.mean(np.hypot(log['x'][tail] - target[0],
                                                log['z'][tail] - target[1])))
    log['theta_rms'] = float(np.sqrt(np.mean(log['theta'][tail] ** 2)))
    # Качество переходного процесса по углу: установление в |theta| < 0.01 рад
    # и перерегулирование (максимальный выброс в сторону, противоположную старту)
    outside = np.nonzero(np.abs(log['theta']) > 0.01)[0]
    log['theta_settle'] = float(log['t'][outside[-1]]) if len(outside) else 0.0
    log['theta_overshoot'] = float(np.max(-np.sign(start['theta']) * log['theta'], initial=0.0))
    return log


def control_freq_experiment(model, data, freq_hz, sim_time=10.0):
    """Эксперимент 4.2: влияние частоты управления (идеальные сенсоры)."""
    return run_episode(model, data, freq_hz=freq_hz, sim_time=sim_time)


def control_delay_experiment(model, data, delay_ms, freq_hz=500, sim_time=10.0):
    """Эксперимент 4.4: влияние задержки измерения при фиксированной частоте."""
    sim_dt = model.opt.timestep
    perception = PerceptionLayer(sim_dt, delay_steps=round(delay_ms / 1000.0 / sim_dt))
    return run_episode(model, data, freq_hz=freq_hz, perception=perception,
                       sim_time=sim_time)
