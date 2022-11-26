from cmath import pi, cos, sin

import numpy as np
from numpy import arctan2

from pedestrian_data import PedestrianLocus


# TODO: 适配更多姿态的测量
@np.vectorize
def calculate_phi_from_gravity(x, y, z):
    return np.arccos((x * 0 + y * 0 + z * 1) / np.sqrt(x ** 2 + y ** 2 + z ** 2))

@np.vectorize
def calculate_theta_from_magnetometer(x, y):
    # 我们将(北-南)看作x轴，手机将长轴作为y轴
    return arctan2(x, y)

def measure_initial_attitude(locus: PedestrianLocus, window_size):
    # NOTICE: 他只能用来计算手持姿态

    gravity_frame = locus.data["Accelerometer"] - locus.data["Linear Acceleration"]
    magnetometer_frame = locus.data["Magnetometer"]

    initial_theta = calculate_theta_from_magnetometer(magnetometer_frame[:window_size, 1].mean(),
                                     magnetometer_frame[:window_size, 2].mean())
    initial_phi = calculate_phi_from_gravity(gravity_frame[:window_size, 1].mean(),
                                 gravity_frame[:window_size, 2].mean(),
                                 gravity_frame[:window_size, 3].mean())

    return initial_theta, initial_phi

def __rotation_x(theta):
    return np.matrix([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])

def __rotation_y(theta):
    return np.matrix([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]])

def __rotation_z(theta):
    return np.matrix([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta), 0],
                      [0, 0, 1]])

CONV_SIZE = 30
def moving_avg(x):
    return np.convolve(x, np.logspace(0.1, 0.5, CONV_SIZE, endpoint=True) /
                               sum(np.logspace(0.1, 0.5, CONV_SIZE, endpoint=True)), mode="same")