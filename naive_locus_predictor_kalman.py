from math import cos, sin, atan2
import numpy as np
from pedestrian_data import PedestrianLocus, PedestrianDataset
from filterpy.kalman import KalmanFilter


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


def __init_kalman() -> KalmanFilter:
    dt = 0.02
    sigma_a = 0.5
    sigma_x = sigma_y = sigma_z = 0.05
    f = KalmanFilter(dim_x=9, dim_z=6)  # z就是前面几个是0，加速度值不是实际值

    # 转换矩阵F
    f.F = np.array([[1., dt, 0.5 * dt * dt, 0, 0, 0, 0, 0, 0],
                    [0, 1, dt, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, dt, 0.5 * dt * dt, 0, 0, 0],
                    [0, 0, 0, 0, 1, dt, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1., dt, 0.5 * dt * dt],
                    [0, 0, 0, 0, 0, 0, 0, 1, dt],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1.]
                    ])

    # 过程噪声
    f.Q = np.array([[0.25 * pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2), 0, 0, 0, 0, 0, 0],
                    [0.5 * pow(dt, 3), dt * dt, dt, 0, 0, 0, 0, 0, 0],
                    [0.5 * pow(dt, 2), dt, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.25 * pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2), 0, 0, 0],
                    [0, 0, 0, 0.5 * pow(dt, 3), dt * dt, dt, 0, 0, 0],
                    [0, 0, 0, 0.5 * pow(dt, 2), dt, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.25 * pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2)],
                    [0, 0, 0, 0, 0, 0, 0.5 * pow(dt, 3), dt * dt, dt],
                    [0, 0, 0, 0, 0, 0, 0.5 * pow(dt, 2), dt, 1]
                    ]) * sigma_a * sigma_a

    # 测量噪声 , 必须是二维数组
    f.R = np.array([[sigma_x * sigma_x, 0, 0, 0, 0, 0],
                    [0, sigma_a * sigma_a, 0, 0, 0, 0],
                    [0, 0, sigma_y * sigma_y, 0, 0, 0],
                    [0, 0, 0, sigma_a * sigma_a, 0, 0],
                    [0, 0, 0, 0, sigma_z * sigma_z, 0],
                    [0, 0, 0, 0, 0, sigma_a * sigma_a]])

    # 观测矩阵
    f.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]
                    ])
    # 初始值
    f.x = np.zeros(9)

    # 协方差矩阵  默认是个单位矩阵 *500就行
    f.P *= sigma_z
    return f


"""
一个朴素的预测模型（对于p的预测还不是很准，但是对于姿态预测不错）
即使不涉及姿态，p仍然不准，比如在桌面上画正方形，加入卡尔曼滤波试试看

@:parameter theta: 从世界坐标系，旋转到当前坐标系的极角（手机头与地磁的夹角）
@:parameter phi: 从世界坐标系，旋转到当前坐标系的仰角（手机头与地平面的夹角）
"""


def predict(locus: PedestrianLocus, theta, phi, no_rotation=False):
    p, v = np.zeros((3, 1)), np.zeros((3, 1))  # 获取一个初态
    x, z = np.zeros((9, 1)), np.zeros((6, 1))
    kf = __init_kalman()

    # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
    # imu_to_earth = __rotation_y(-phi) @ __rotation_z(-theta)\
    imu_to_earth = __rotation_z(theta) @ __rotation_y(phi)

    gyroscope_imu_frame = locus.data["Gyroscope"][:, 1:]
    acceleration_imu_frame = locus.data["Linear Acceleration"][:, 1:]
    time_frame = locus.data["Gyroscope"][:, 0]

    thetas, phis = np.empty(len(time_frame) - 2), np.empty(len(time_frame) - 2)
    positions, speeds, accelerations = np.empty((len(time_frame) - 2, 3)), \
                                       np.empty((len(time_frame) - 2, 3)), \
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

        # TODO 卡尔曼修正
        z[1] = acceleration_earth[0]
        z[3] = acceleration_earth[1]
        z[5] = acceleration_earth[2]
        z = z.reshape(-1, 1)
        kf.update(z)
        kf.predict()

        # p += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        # v += acceleration_earth * delta_t

        imu_to_earth = imu_to_earth @ rotation

        p = kf.x[[0, 3, 6]].reshape(-1, 1)
        v = kf.x[[1, 4, 7]].reshape(-1, 1)

        # 做记录
        thetas[index] = atan2(-imu_to_earth[0, 1], imu_to_earth[1, 1])
        phis[index] = atan2(-imu_to_earth[2, 0], imu_to_earth[2, 2])
        positions[index] = p.T
        speeds[index] = v.T
        accelerations[index] = acceleration_earth.T

    return positions, speeds, accelerations, thetas, phis


if __name__ == "__main__":
    pass
