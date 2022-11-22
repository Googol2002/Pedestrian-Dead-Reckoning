from math import cos, sin

import numpy as np

from pedestrian_data import PedestrianLocus, PedestrianDataset


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

def predict(locus: PedestrianLocus, theta, phi):
    x, v = np.zeros((3, 1)), np.zeros((3, 1))   # 获取一个初态

    # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
    imu_to_earth = __rotation_y(-phi) @ __rotation_z(theta)

    gyroscope_imu_frame = locus.data["Gyroscope"][:, 1:]
    acceleration_imu_frame = locus.data["Linear Acceleration"][:, 1:]
    time_frame = locus.data["Gyroscope"][:, 0]
    for index, (gyroscope_imu, acceleration_imu) in enumerate(
            zip(gyroscope_imu_frame[1: -1], acceleration_imu_frame[1: -1])):
        index += 1  # 用原始表中的index
        delta_t = (time_frame[index + 1] - time_frame[index - 1]) / 2

        # 姿态变化
        rotation = __rotation_x(delta_t * gyroscope_imu[0]) \
            @ __rotation_y(delta_t * gyroscope_imu[1]) \
            @ __rotation_z(delta_t * gyroscope_imu[2])

        # 牛顿力学
        acceleration_earth = imu_to_earth @ np.expand_dims(acceleration_imu, 1)
        x += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        v += acceleration_earth * delta_t

        imu_to_earth = imu_to_earth @ __rotation_x(delta_t * gyroscope_imu[0])\
                       @ __rotation_y(delta_t * gyroscope_imu[1])\
                       @ __rotation_z(delta_t * gyroscope_imu[2])

    return x, v, imu_to_earth


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"])
    predict(dataset["NorthEastSouthWest"], 0, 0)
