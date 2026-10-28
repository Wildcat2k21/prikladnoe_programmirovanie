"""
Слой Perception (Восприятие) — 2D-квадрокоптер
================================================
Извлечение состояния из симуляции, добавление шума и задержки.

Сенсоры борта:
- IMU (гироскоп + акселерометр): угол pitch и угловая скорость, частота физики;
- высотомер (барометр/дальномер): только высота z, вертикальная скорость
  оценивается как отфильтрованная производная высоты;
- позиционный сенсор (камера/оптический поток): x и x_dot, может работать
  на пониженной частоте (например, 30 fps) — рассинхронизация потоков.

Демонстрирует:
- шумные, неполные и непрерывно меняющиеся входные данные;
- рассинхронизацию потоков данных (камера 30 fps vs IMU 1000 Гц).
"""

import numpy as np
from collections import deque


def noisy_sensor(true_value, sigma=0.02, rng=np.random):
    """Аддитивный гауссовский шум (п. 2.2 методички)."""
    return true_value + rng.normal(0, sigma)


class DelayedSensor:
    """
    Задержка измерения на delay_steps шагов (п. 2.3 методички).

    Отличия от листинга в методичке:
    - delay_steps=0 допустим (в методичке deque(maxlen=0) падает);
    - буфер заполняется первым измерением, а не нулями — иначе в начале
      эпизода высотомер «видит» z=0 и контроллер даёт рывок тяги.
    """

    def __init__(self, delay_steps):
        self.delay_steps = delay_steps
        self.buffer = deque(maxlen=delay_steps + 1)

    def read(self, value):
        if not self.buffer:
            self.buffer.extend([value] * (self.delay_steps + 1))
        self.buffer.append(value)
        return self.buffer[0]


class SampleAndHold:
    """Сенсор с собственной частотой: между отсчётами выдаёт последнее значение."""

    def __init__(self, period_steps):
        self.period_steps = max(1, period_steps)
        self.counter = 0
        self.value = None

    def read(self, value):
        if self.value is None or self.counter % self.period_steps == 0:
            self.value = value
        self.counter += 1
        return self.value


class PerceptionLayer:
    """
    Полный слой восприятия 2D-квадрокоптера.

    Вызывается на каждом шаге физики (сенсоры работают непрерывно),
    контроллер забирает последнее доступное наблюдение на своей частоте.

    Возвращает dict с ключами x, z, theta, x_dot, z_dot, theta_dot
    (измеренные) и *_true (истинные, только для графиков).
    """

    def __init__(self, sim_dt, delay_steps=0, sigma_imu=0.0, sigma_alt=0.0,
                 sigma_pos=0.0, pos_rate_hz=None, z_dot_tau=0.02, seed=0):
        """
        Args:
            sim_dt: шаг физики, с
            delay_steps: задержка всех каналов в шагах физики
            sigma_imu: шум IMU (рад, рад/с)
            sigma_alt: шум высотомера (м)
            sigma_pos: шум позиционного сенсора (м, м/с)
            pos_rate_hz: частота позиционного сенсора (None = частота физики)
            z_dot_tau: постоянная времени фильтра производной высоты, с
            seed: зерно генератора шума (воспроизводимость экспериментов)
        """
        self.sim_dt = sim_dt
        self.delay_steps = delay_steps
        self.sigma_imu = sigma_imu
        self.sigma_alt = sigma_alt
        self.sigma_pos = sigma_pos
        self.rng = np.random.default_rng(seed)

        # Отказ высотомера: с момента fault_time шум становится fault_sigma
        self.fault_time = None
        self.fault_sigma = 0.0

        self.channels = ('x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot')
        self.delays = {name: DelayedSensor(delay_steps) for name in self.channels}

        period = 1 if pos_rate_hz is None else round(1.0 / pos_rate_hz / sim_dt)
        self.pos_hold = {name: SampleAndHold(period) for name in ('x', 'x_dot')}

        # Оценка z_dot: производная высоты через фильтр 1-го порядка
        self.alpha = sim_dt / max(z_dot_tau, sim_dt)
        self.z_prev = None
        self.z_dot_est = 0.0

    def inject_altimeter_fault(self, fault_time, fault_sigma):
        """Запланировать отказ высотомера (резкий рост шума)."""
        self.fault_time = fault_time
        self.fault_sigma = fault_sigma

    def altimeter_sigma(self, t):
        if self.fault_time is not None and t >= self.fault_time:
            return self.fault_sigma
        return self.sigma_alt

    def observe(self, data):
        """
        Извлечь состояние из MjData и добавить шум/задержку.

        Args:
            data: объект mujoco.MjData

        Returns:
            dict с измеренными и истинными значениями
        """
        x_true, z_true, theta_true = data.qpos[0], data.qpos[1], data.qpos[2]
        x_dot_true, z_dot_true, theta_dot_true = data.qvel[0], data.qvel[1], data.qvel[2]

        # IMU — на частоте физики
        theta = noisy_sensor(theta_true, self.sigma_imu, self.rng)
        theta_dot = noisy_sensor(theta_dot_true, self.sigma_imu, self.rng)

        # Высотомер меряет только z; z_dot — отфильтрованная производная
        z = noisy_sensor(z_true, self.altimeter_sigma(data.time), self.rng)
        if self.z_prev is not None:
            z_dot_raw = (z - self.z_prev) / self.sim_dt
            self.z_dot_est += self.alpha * (z_dot_raw - self.z_dot_est)
        self.z_prev = z
        z_dot = self.z_dot_est

        # Позиционный сенсор — возможно, на пониженной частоте
        x = self.pos_hold['x'].read(noisy_sensor(x_true, self.sigma_pos, self.rng))
        x_dot = self.pos_hold['x_dot'].read(noisy_sensor(x_dot_true, self.sigma_pos, self.rng))

        measured = {'x': x, 'z': z, 'theta': theta,
                    'x_dot': x_dot, 'z_dot': z_dot, 'theta_dot': theta_dot}
        obs = {name: self.delays[name].read(measured[name]) for name in self.channels}

        obs.update({
            'x_true': x_true, 'z_true': z_true, 'theta_true': theta_true,
            'x_dot_true': x_dot_true, 'z_dot_true': z_dot_true,
            'theta_dot_true': theta_dot_true,
            'z_raw': z,  # зашумлённое, но не задержанное (для графика 2.4)
        })
        return obs
