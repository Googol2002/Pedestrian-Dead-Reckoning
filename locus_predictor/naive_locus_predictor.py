from math import cos, sin, atan2
from numpy import arctan2, pi

import numpy as np

from locus_predictor.helper import measure_initial_attitude
from pedestrian_data import PedestrianLocus, PedestrianDataset
from scipy.signal import find_peaks


@np.vectorize
def calculate_theta(x, y):
    # 我们将(北-南)看作x轴，手机将长轴作为y轴
    return pi / 2 - arctan2(y, x)


def measure_attitude(locus: PedestrianLocus):
    gravity_frame = locus.data["Accelerometer"] - locus.data["Linear Acceleration"]
    magnetometer_frame = locus.data["Magnetometer"]
    gyroscope_frame = locus.data["Gyroscope"]

    # 我们要做一个MA降噪
    CONV_SIZE = 30
    moving_avg = lambda x: np.convolve(x, np.logspace(0.1, 0.5, 30, endpoint=True) /
                               sum(np.logspace(0.1, 0.5, 30, endpoint=True)), mode="same")
    measured_theta = calculate_theta(magnetometer_frame[:, 1], magnetometer_frame[:, 2])
    # measured_phi = calculate_phi(ma(gravity_frame[:, 1]), ma(gravity_frame[:, 2]), ma(gravity_frame[:, 3]))

    # return measured_theta, measured_phi


# 50Hz 我们假设，人1s内不会迈太多步
MIN_PERIOD = 20
PROMINENCE = (0.05, None)
# 假设步幅为0.8m
PACE_STEP = 0.8

CONV_SIZE = 30
moving_avg = lambda x: np.convolve(x, np.logspace(0.1, 0.5, 30, endpoint=True)/ sum(np.logspace(0.1, 0.5, 30, endpoint=True)), mode="same")

"""
一个朴素的预测模型（对于p的预测还不是很准，但是对于姿态预测不错）
即使不涉及姿态，p仍然不准，比如在桌面上画正方形，加入卡尔曼滤波试试看

@:parameter theta: 从世界坐标系，旋转到当前坐标系的极角（手机头与地磁的夹角）
@:parameter phi: 从世界坐标系，旋转到当前坐标系的仰角（手机头与地平面的夹角）
@:parameter no_rotation：默认存在旋转为False，当设置为True后，将不再考虑姿态矫正
@:parameter walk: 假设人走路时，加速度方向与人移动方向相同，默认为None；
            True：则会开启假设，并将positions改用步幅计算
            False：则会关闭假设，并将positions改用步幅计算
            None：利用牛顿力学计算positions
            
@:return positions: 不保证和时间对齐
@:return properties: 一些属性
"""
def predict(locus: PedestrianLocus, attitude=None, degree=0, walk=True):
    p, v = np.zeros((3, 1)), np.zeros((3, 1))   # 获取一个初态
    theta, phi = attitude if attitude else measure_initial_attitude(locus, 30)

    # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
    imu_to_earth = __rotation_z(theta) @ __rotation_y(phi)

    gyroscope_imu_frame = locus.data["Gyroscope"][:, 1:]
    magnetometer_imu_frame = locus.data["Magnetometer"][:, 1:]
    acceleration_imu_frame = locus.data["Linear Acceleration"][:, 1:]
    time_frame = locus.data["Gyroscope"][:, 0]

    thetas, phis, alphas, directions = [np.empty(len(time_frame) - 2) for _ in range(4)]
    positions, speeds, accelerations = [np.empty((len(time_frame) - 2, 3)) for _ in range(3)]

    for index, (gyroscope_imu, acceleration_imu, magnetometer_imu) in enumerate(
            zip(gyroscope_imu_frame[1: -1], acceleration_imu_frame[1: -1], magnetometer_imu_frame[1: -1])):
        delta_t = (time_frame[index + 2] - time_frame[index]) / 2

        # 姿态变化
        rotation = __rotation_z(delta_t * gyroscope_imu[2]) @\
            __rotation_y(delta_t * gyroscope_imu[1]) @ \
            __rotation_x(delta_t * gyroscope_imu[0])
        imu_to_earth = imu_to_earth @ rotation

        # 计算姿态角
        sy = np.sqrt(imu_to_earth[2, 2] ** 2 + imu_to_earth[2, 1] ** 2)
        singular = sy < 1e-3
        if not singular:
            thetas[index] = atan2(imu_to_earth[1, 0], imu_to_earth[0, 0])
            phis[index] = atan2(-imu_to_earth[2, 0], sy)
            alphas[index] = atan2(imu_to_earth[2, 1], imu_to_earth[2, 2])
        else:
            thetas[index] = 0
            phis[index] = atan2(-imu_to_earth[2, 0], sy)
            alphas[index] = atan2(-imu_to_earth[1, 2], imu_to_earth[1, 1])

        # 这种计算正确吗？

        # 与磁力计较准极角，alpha滤波
        # magnetometer_earth = imu_to_earth @ magnetometer_imu
        # measured_theta = calculate_theta(magnetometer_earth[0, 0], magnetometer_earth[0, 1])
        # thetas[index] = thetas[index] + 0.01 * (measured_theta - thetas[index])
        imu_to_earth = __rotation_z(thetas[index]) @ __rotation_y(phis[index]) @ __rotation_x(alphas[index])

        # 牛顿力学
        acceleration_earth = imu_to_earth @ np.expand_dims(acceleration_imu, 1)
        p += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        v += acceleration_earth * delta_t

        # 利用牛顿力学计算p
        positions[index] = p.T
        speeds[index] = v.T
        accelerations[index] = acceleration_earth.T
        # directions[index] = (acceleration_earth[0: 2] / np.sqrt(acceleration_earth[0]
        #                                                         ** 2 + acceleration_earth[1] ** 2)).T
        directions[index] = thetas[index]

    peaks_index, _ = find_peaks(speeds[:, 2], distance=MIN_PERIOD, prominence=PROMINENCE)

    # 步幅步频
    walk_positions, walk_directions = None, None
    if walk:
        walk_positions = np.zeros((len(peaks_index) + 1, 2))
        walk_directions = np.zeros(len(peaks_index))
        p = np.zeros(2)

        for index, peak in enumerate(peaks_index):
            # direction = directions[peak - 2: peak + 2].mean(axis=0)   # 只要xOy上方向
            direction = directions[peak]
            walk_directions[index] = direction
            p += PACE_STEP * np.asarray([cos(direction), sin(direction)])
            walk_positions[index + 1] = p

    # 汇总数据
    return positions[:, :2], {"speeds": speeds, "accelerations": accelerations,
                              "thetas": thetas, "phis": phis, "time": time_frame[1:-1],
                              "peaks": peaks_index, "walk_positions": walk_positions,
                              "walk_directions": walk_directions,
                              "walk_time": time_frame[1 + peaks_index]}


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


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"], gps_preprocessed=False)
    predict(dataset["随机漫步1"], 0, 0, walk=True)
    # measure_attitude(dataset["水平面(北东南西，加入扰动)"])