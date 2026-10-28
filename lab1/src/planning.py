"""
Слой Planning (Планирование) — 2D-квадрокоптер
================================================
Вариант №10: висение в точке, PD по высоте.

Каскадная схема:
    1. PD по высоте      z  -> суммарная тяга T            (основной контур варианта)
    2. PD по положению   x  -> желаемый угол theta_des     (медленный внешний контур)
    3. PD по углу    theta  -> момент tau                  (быстрый внутренний контур)
    4. Микшер (T, tau) -> тяги левого/правого моторов

Демонстрирует:
- численный решатель вместо if-else: управление — это решение уравнений
  динамики m*z'' = T*cos(theta) - m*g и I*theta'' = tau;
- жёсткую траекторию по waypoints (п. 3.3, Atlas) против перестраиваемой цели.
"""

import numpy as np

G = 9.81


def quadrotor_params(model):
    """Масса, момент инерции по pitch и плечо моторов из MjModel."""
    body = model.body('quadrotor').id
    mass = model.body_subtreemass[body]
    left, right = model.body('left_motor').id, model.body('right_motor').id
    arm = abs(model.body_pos[right][0])
    # I_yy рамы + моторы (собственная инерция + m*r^2)
    inertia = model.body_inertia[body][1]
    for motor in (left, right):
        inertia += model.body_inertia[motor][1] + model.body_mass[motor] * arm ** 2
    return mass, inertia, arm


class HoverPDPlanner:
    """
    Каскадный PD-регулятор висения в точке (x_target, z_target).

    PD по высоте с компенсацией гравитации и наклона:
        T = m * (g + Kp_z*(z* - z) - Kd_z*z_dot) / cos(theta)
    """

    def __init__(self, mass, inertia, arm,
                 Kp_z=25.0, Kd_z=10.0,
                 Kp_x=1.0, Kd_x=1.5, theta_max=0.4,
                 Kp_theta=1600.0, Kd_theta=56.0,
                 thrust_max=10.0):
        """
        Args:
            mass, inertia, arm: параметры аппарата (см. quadrotor_params)
            Kp_z, Kd_z: PD по высоте (w ~ 5 рад/с, zeta = 1)
            Kp_x, Kd_x: PD по горизонтали -> желаемый угол
            theta_max: ограничение желаемого угла, рад
            Kp_theta, Kd_theta: PD по углу (w ~ 40 рад/с, zeta ~ 0.7)
            thrust_max: максимальная тяга одного мотора, Н
        """
        self.mass, self.inertia, self.arm = mass, inertia, arm
        self.Kp_z, self.Kd_z = Kp_z, Kd_z
        self.Kp_x, self.Kd_x = Kp_x, Kd_x
        self.theta_max = theta_max
        self.Kp_theta, self.Kd_theta = Kp_theta, Kd_theta
        self.thrust_max = thrust_max

    def hover_thrust(self):
        """Тяга одного мотора для висения, Н."""
        return self.mass * G / 2

    def compute(self, obs, target):
        """
        Вычислить тяги моторов.

        Args:
            obs: dict из PerceptionLayer (или FailOperationalEstimator)
            target: (x_target, z_target)

        Returns:
            (np.array([F_left, F_right]), T_total): тяги моторов и суммарная тяга
        """
        x_t, z_t = target
        theta = obs['theta']

        # Внешний контур по x: наклон theta>0 даёт ускорение в +x
        theta_des = self.Kp_x * (x_t - obs['x']) - self.Kd_x * obs['x_dot']
        theta_des = np.clip(theta_des, -self.theta_max, self.theta_max)

        # PD по высоте (суть варианта №10)
        a_z = self.Kp_z * (z_t - obs['z']) - self.Kd_z * obs['z_dot']
        thrust = self.mass * (G + a_z) / max(np.cos(theta), 0.3)

        # Внутренний контур по углу
        tau = self.inertia * (self.Kp_theta * (theta_des - theta)
                              - self.Kd_theta * obs['theta_dot'])

        # Микшер: тяга правого мотора даёт отрицательный момент по pitch
        f_left = thrust / 2 + tau / (2 * self.arm)
        f_right = thrust / 2 - tau / (2 * self.arm)
        forces = np.clip([f_left, f_right], 0.0, self.thrust_max)
        return forces, forces.sum()


class WaypointPlanner:
    """
    Жёсткая траектория по waypoints (п. 3.3, вариант C).

    Траектория задаётся до старта и не перестраивается — как у Atlas.
    Переход к следующей точке происходит при попадании в радиус tolerance.
    """

    def __init__(self, waypoints, tolerance=0.05):
        """
        Args:
            waypoints: list of (x, z)
            tolerance: радиус достижения точки, м
        """
        self.waypoints = list(waypoints)
        self.tolerance = tolerance
        self.current_idx = 0

    def get_target(self, x, z):
        """Текущая целевая точка и флаг «маршрут пройден»."""
        target = self.waypoints[self.current_idx]
        if np.hypot(x - target[0], z - target[1]) < self.tolerance \
                and self.current_idx < len(self.waypoints) - 1:
            self.current_idx += 1
            target = self.waypoints[self.current_idx]
        return target, self.current_idx == len(self.waypoints) - 1
