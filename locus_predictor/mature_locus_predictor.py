import numpy as np
from numpy import cos, sin

from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from locus_predictor.helper import measure_initial_attitude, measure_initial_attitude_advanced
from pedestrian_data import PedestrianLocus, PedestrianDataset

# 50Hz 我们假设，人1s内不会迈太多步
MIN_PERIOD = 20
PROMINENCE = (0.05, None)
# 假设步幅为0.8m
PACE_STEP = 0.8


def locus_predictor(attitude=None, walk_direction_bias=0, pace_inference=None):
    """
    一个朴素的预测模型（对于p的预测还不是很准，但是对于姿态预测不错）
    即使不涉及姿态，p仍然不准，比如在桌面上画正方形，加入卡尔曼滤波试试看

    :param attitude: (theta, phi) 从世界坐标系，旋转到当前坐标系的极角（手机头与地磁的夹角），
    旋转到当前坐标系的滚角（手机头与地平面的夹角）
    :param walk_direction_bias: 手动偏移
    :param pace_inference:步幅推断器
    :return:一个预测器
    """
    def predict(locus: PedestrianLocus):
        theta, phi = attitude if attitude else measure_initial_attitude(locus, 30)
        # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
        imu_to_earth = Rotation.from_euler("ZYX", [theta, 0, phi])
        # imu_to_earth = measure_initial_attitude_advanced(locus, 30)

        # 提取传感器信息
        gyroscope_imu_frame, magnetometer_imu_frame, acceleration_imu_frame = locus.data["Gyroscope"][:, 1:], \
            locus.data["Magnetometer"][:, 1:], locus.data["Linear Acceleration"][:, 1:]
        time_frame = locus.data["Gyroscope"][:, 0]

        # 记录力学信息
        info = __record_movement(locus, imu_to_earth, gyroscope_imu_frame,
                                 magnetometer_imu_frame, acceleration_imu_frame, time_frame,walk_direction_bias)

        inference = pace_inference(info) if pace_inference else lambda x, y: PACE_STEP
        # 模拟走路
        walk_positions, walk_directions = __simulated_walk(locus, info, inference, walk_direction_bias)

        # 插值
        return __aligned_with_gps(locus, info, walk_positions, walk_directions), info

    return predict


def __record_movement(locus, imu_to_earth, gyroscope_imu_frame,
                      magnetometer_imu_frame, acceleration_imu_frame, time_frame,walk_direction_bias):
    p, v = np.zeros(3), np.zeros(3)  # 获取一个初态

    thetas, phis, alphas, directions = [np.empty(len(time_frame) - 2) for _ in range(4)]
    speeds, accelerations, y_directions = [np.empty((len(time_frame) - 2, 3)) for _ in range(3)]
    for index, (gyroscope_imu, acceleration_imu, magnetometer_imu) in enumerate(
            zip(gyroscope_imu_frame[1: -1], acceleration_imu_frame[1: -1], magnetometer_imu_frame[1: -1])):
        delta_t = (time_frame[index + 2] - time_frame[index]) / 2
        # 姿态变化
        imu_to_earth = imu_to_earth * Rotation.from_euler("xyz", delta_t * gyroscope_imu)

        # 计算姿态角
        # thetas[index], phis[index], alphas[index] = imu_to_earth.as_euler("ZYX")
        # 可以解决摆手机问题
        thetas[index], alphas[index], phis[index] = imu_to_earth.as_euler("ZXY")
        # y_directions[index] = imu_to_earth.apply(np.array([0, 1, 0]))

        # 牛顿力学
        acceleration_earth = imu_to_earth.apply(acceleration_imu)
        p += v * delta_t + acceleration_earth * (delta_t ** 2) / 2
        v += acceleration_earth * delta_t

        # 记录运动学信息
        speeds[index] = v
        accelerations[index] = acceleration_earth
        directions[index] = thetas[index]

    peaks_index, _ = find_peaks(speeds[:, 2], distance=MIN_PERIOD, prominence=PROMINENCE)

    info = {"speeds": speeds, "accelerations": accelerations,
            "thetas": thetas, "phis": phis, "time": time_frame[1:-1],
            "peaks": peaks_index, "directions": directions,
            "walk_time": time_frame[1 + peaks_index], "locus": locus,"walk_direction_bias":walk_direction_bias}

    return info


def __simulated_walk(locus, info, inference, walk_direction_bias):
    peaks_index = info["peaks"]
    directions = info["directions"]

    walk_positions = np.zeros((len(peaks_index) + 1, 2))
    walk_directions = np.zeros(len(peaks_index))
    p = np.zeros(2)
    for index, peak in enumerate(peaks_index):
        direction = directions[peak-2: peak+3].mean()
        walk_directions[index] = direction + walk_direction_bias
        pace = inference(index, peak)
        p += pace * np.asarray([cos(np.pi / 2 + walk_directions[index]), sin(np.pi / 2 + walk_directions[index])])
        # y_direction = y_directions[peak][:2]
        # p += pace * y_direction / np.sqrt(y_direction[0] ** 2 + y_direction[1] ** 2)
        walk_positions[index + 1] = p

    # GPS整体反馈矫正
    positions,_ = __aligned_with_gps(locus, info, walk_positions, walk_directions)
    positions=positions.T
    start_to_end_positions = np.linalg.norm(np.array([positions[0][-1], positions[1][-1]]) - np.array([positions[0][0], positions[1][0]]))
    Lati = locus.relative_location["relative_x (m)"].to_numpy()
    Longi = locus.relative_location["relative_y (m)"].to_numpy()
    start_to_end_GPS = np.linalg.norm(np.array([Lati[-1], Longi[-1]]) - np.array([Lati[0], Longi[0]]))
    #print("Compare before correction:",start_to_end_positions,start_to_end_GPS)
    transform=start_to_end_GPS/start_to_end_positions#pace整体乘一个比例
    #print(transform)

    #重新算一遍
    walk_positions = np.zeros((len(peaks_index) + 1, 2))
    walk_directions = np.zeros(len(peaks_index))
    p = np.zeros(2)
    for index, peak in enumerate(peaks_index):
        direction = directions[peak-2: peak+3].mean()
        walk_directions[index] = direction + walk_direction_bias
        pace = inference(index, peak)*transform
        p += pace * np.asarray([cos(np.pi / 2 + walk_directions[index]), sin(np.pi / 2 + walk_directions[index])])
        walk_positions[index + 1] = p


    info["walk_positions"] = walk_positions
    info["walk_directions"] = walk_directions

    return walk_positions, walk_directions


def __aligned_with_gps(locus, info, walk_positions, walk_directions):
    peaks_index = info["peaks"]
    walk_time = info["walk_time"]

    # 插值
    if len(peaks_index) > 3:
        positions = interp1d(np.concatenate((np.array([0]), walk_time)),
                             walk_positions, kind='cubic', axis=0, fill_value="extrapolate")\
            (locus.y_frame["location_time"])
        directions = interp1d(walk_time, walk_directions, kind='cubic', axis=0, fill_value="extrapolate")\
            (locus.y_frame["location_time"])

    else:
        positions = None
        directions = None

    return positions, directions


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"])
    locus_predictor = locus_predictor()
    #predict(dataset["NorthEastSouthWest"])
