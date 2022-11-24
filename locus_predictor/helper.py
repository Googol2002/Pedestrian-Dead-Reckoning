from cmath import pi

import numpy as np
from numpy import arctan2

from pedestrian_data import PedestrianLocus


def measure_initial_attitude(locus: PedestrianLocus, window_size):
    # NOTICE: 他只能用来计算手持姿态
    # TODO: 适配更多姿态的测量
    @np.vectorize
    def calculate_phi(x, y, z):
        return np.arccos((x * 0 + y * 0 + z * 1) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

    @np.vectorize
    def calculate_theta(x, y):
        # 我们将(北-南)看作x轴，手机将长轴作为y轴
        return pi / 2 - arctan2(y, x)

    gravity_frame = locus.data["Accelerometer"] - locus.data["Linear Acceleration"]
    magnetometer_frame = locus.data["Magnetometer"]

    initial_theta = calculate_theta(magnetometer_frame[:window_size, 1].mean(),
                                     magnetometer_frame[:window_size, 2].mean())
    initial_phi = calculate_phi(gravity_frame[:window_size, 1].mean(),
                                 gravity_frame[:window_size, 2].mean(),
                                 gravity_frame[:window_size, 3].mean())

    return initial_theta, initial_phi