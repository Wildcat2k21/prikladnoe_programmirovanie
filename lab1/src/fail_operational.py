"""
Fail-Operational (Отказоустойчивость) — 2D-квадрокоптер
=========================================================
Детектор отказа высотомера и переключение на резервный источник —
предсказание высоты по модели динамики.

Демонстрирует:
- Fail-Operational: при отказе сенсора система деградирует, но летит;
- Fail-Safe (без детектора): шум отказавшего высотомера проходит через
  производную в PD по высоте и роняет аппарат.
"""

import numpy as np
from collections import deque

from planning import G


class SensorFailureDetector:
    """
    Детектор отказа по дисперсии сигнала (п. 5.2 методички).

    Если дисперсия измерений за последние window_size шагов превышает
    threshold — сенсор считается отказавшим.
    """

    def __init__(self, window_size=50, threshold=1e-3):
        """
        Args:
            window_size: окно, шагов физики (50 шагов = 50 мс)
            threshold: порог дисперсии, м^2. Номинальный шум высотомера
                sigma=0.002 м даёт ~4e-6, манёвр 0.5 м/с за окно — ~2e-4,
                отказ sigma=0.1 м — ~1e-2.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = deque(maxlen=window_size)
        self.failed = False
        self.variance = 0.0

    def update(self, value):
        """Добавить измерение; вернуть True, если сенсор признан отказавшим."""
        self.buffer.append(value)
        if len(self.buffer) == self.window_size:
            self.variance = float(np.var(self.buffer))
            self.failed = self.variance > self.threshold
        return self.failed


class ModelPredictor:
    """
    Резервный источник высоты — наблюдатель на модели динамики:

        z'' = T * cos(theta) / m - g

    T — командная тяга (известна бортовому компьютеру), theta — от IMU
    (исправен). Пока высотомер исправен, оценка корректируется по нему;
    при отказе — чистое предсказание по модели (dead reckoning).
    """

    def __init__(self, sim_dt, mass, omega=20.0, innovation_gate=0.03):
        """
        Args:
            sim_dt: шаг физики, с
            mass: масса, которую «знает» модель (можно задать с ошибкой)
            omega: полоса коррекции наблюдателя, рад/с
            innovation_gate: невязки больше этого (м) не используются для
                коррекции — защита от выбросов в первые мс отказа
        """
        self.dt = sim_dt
        self.mass = mass
        self.l1, self.l2 = 2 * omega, omega ** 2
        self.gate = innovation_gate
        self.z = None
        self.z_dot = 0.0

    def update(self, z_meas, theta, thrust, use_measurement):
        if self.z is None:
            self.z = z_meas
        a = thrust * np.cos(theta) / self.mass - G
        e = z_meas - self.z
        if not use_measurement or abs(e) > self.gate:
            e = 0.0
        self.z += self.dt * (self.z_dot + self.l1 * e)
        self.z_dot += self.dt * (a + self.l2 * e)
        return self.z, self.z_dot


class FailOperationalEstimator:
    """
    Прослойка между Perception и Planning.

    Высотомер исправен -> в Planning идут измерения как есть
    (поведение совпадает с режимом без детектора).
    Отказ детектирован -> z и z_dot подменяются предсказанием модели.
    """

    def __init__(self, sim_dt, mass, detector_window=50, detector_threshold=1e-3,
                 model_mass_scale=1.0):
        """
        Args:
            sim_dt: шаг физики, с
            mass: истинная масса аппарата
            detector_window, detector_threshold: параметры детектора
            model_mass_scale: ошибка модели (1.02 = модель считает массу на 2% больше)
        """
        self.detector = SensorFailureDetector(detector_window, detector_threshold)
        self.predictor = ModelPredictor(sim_dt, mass * model_mass_scale)

    def update(self, obs, thrust):
        """
        Args:
            obs: dict из PerceptionLayer
            thrust: суммарная тяга, приложенная на этом шаге, Н

        Returns:
            (obs для Planning, флаг отказа)
        """
        failed = self.detector.update(obs['z'])
        z_hat, z_dot_hat = self.predictor.update(obs['z'], obs['theta'], thrust,
                                                 use_measurement=not failed)
        if not failed:
            return obs, False
        used = dict(obs)
        used['z'], used['z_dot'] = z_hat, z_dot_hat
        return used, True
