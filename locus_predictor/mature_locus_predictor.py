from numpy import cos, sin
from scipy.spatial.transform import Rotation

import numpy as np

from locus_predictor.helper import measure_initial_attitude
from pedestrian_data import PedestrianLocus, PedestrianDataset
from scipy.signal import find_peaks


# 50Hz 我们假设，人1s内不会迈太多步
MIN_PERIOD = 20
PROMINENCE = (0.05, None)
# 假设步幅为0.8m
PACE_STEP = 0.8

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
    p, v = np.zeros(3), np.zeros(3)   # 获取一个初态
    theta, phi = attitude if attitude else measure_initial_attitude(locus, 30)

    # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
    imu_to_earth = Rotation.from_euler("ZYX", [theta, phi, 0])

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
        imu_to_earth = imu_to_earth * Rotation.from_euler("XYZ", delta_t * gyroscope_imu)
        # Rotation.from_quat(np.concatenate((np.asarray([1]), delta_t * gyroscope_imu / 2)))

        # 计算姿态角
        thetas[index], phis[index], alphas[index] = imu_to_earth.as_euler("ZYX")

        # 牛顿力学
        acceleration_earth = imu_to_earth.apply(acceleration_imu)
        p += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        v += acceleration_earth * delta_t

        # 利用牛顿力学计算p
        positions[index] = p
        speeds[index] = v
        accelerations[index] = acceleration_earth
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


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"], gps_preprocessed=False)
    predict(dataset["随机漫步1"], walk=True)
