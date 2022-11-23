from math import cos, sin, atan2

import numpy as np

from pedestrian_data import PedestrianLocus, PedestrianDataset
from scipy.signal import find_peaks


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


# 50Hz 我们假设，人1s内不会迈太多步
MIN_PERIOD = 20
PROMINENCE = (0.05, None)

"""
一个朴素的预测模型（对于p的预测还不是很准，但是对于姿态预测不错）
即使不涉及姿态，p仍然不准，比如在桌面上画正方形，加入卡尔曼滤波试试看

@:parameter theta: 从世界坐标系，旋转到当前坐标系的极角（手机头与地磁的夹角）
@:parameter phi: 从世界坐标系，旋转到当前坐标系的仰角（手机头与地平面的夹角）
"""
def predict(locus: PedestrianLocus, theta, phi, no_rotation=False):
    p, v = np.zeros((3, 1)), np.zeros((3, 1))   # 获取一个初态

    # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
    # imu_to_earth = __rotation_y(-phi) @ __rotation_z(-theta)\
    imu_to_earth = __rotation_z(theta) @ __rotation_y(phi)

    gyroscope_imu_frame = locus.data["Gyroscope"][:, 1:]
    acceleration_imu_frame = locus.data["Linear Acceleration"][:, 1:]
    time_frame = locus.data["Gyroscope"][:, 0]

    thetas, phis = np.empty(len(time_frame) - 2), np.empty(len(time_frame) - 2)
    positions, speeds, accelerations = np.empty((len(time_frame) - 2, 3)),\
                                       np.empty((len(time_frame) - 2, 3)),\
                                       np.empty((len(time_frame) - 2, 3))

    for index, (gyroscope_imu, acceleration_imu) in enumerate(
            zip(gyroscope_imu_frame[1: -1], acceleration_imu_frame[1: -1])):
        delta_t = (time_frame[index + 2] - time_frame[index]) / 2

        # 姿态变化
        if not no_rotation:
            rotation = __rotation_x(delta_t * gyroscope_imu[0]) \
                @ __rotation_y(delta_t * gyroscope_imu[1]) \
                @ __rotation_z(delta_t * gyroscope_imu[2])
        else:
            rotation = np.eye(3)

        # 牛顿力学
        acceleration_earth = imu_to_earth @ np.expand_dims(acceleration_imu, 1)
        p += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        v += acceleration_earth * delta_t

        imu_to_earth = imu_to_earth @ rotation

        # 做记录
        thetas[index] = atan2(-imu_to_earth[0, 1], imu_to_earth[1, 1])
        phis[index] = atan2(-imu_to_earth[2, 0], imu_to_earth[2, 2])
        positions[index] = p.T
        speeds[index] = v.T
        accelerations[index] = acceleration_earth.T

    peaks_index, _ = find_peaks(speeds[:, 2], distance=MIN_PERIOD, prominence=PROMINENCE)

    return positions, speeds, accelerations, thetas, phis, time_frame[1:-1], peaks_index


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"], gps_preprocessed=False)
    predict(dataset["NorthEastSouthWest"], 0, 0)
