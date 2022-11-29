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
Magic_A = 0.37
Magic_B = 0.000155
Magic_C = 0.1638


def transform_x(locus, r):
    x = r[0][:, 0]
    x *= -1
    r[0][:, 0] = x
    return r


def locus_predictor(attitude=None, walk_direction_bias=0,
                    magic=None, pace_inference=None, transform=None, euler="ZXY"):
    """
    一个朴素的预测模型（对于p的预测还不是很准，但是对于姿态预测不错）
    即使不涉及姿态，p仍然不准，比如在桌面上画正方形，加入卡尔曼滤波试试看

    :param euler:
    :param attitude: (theta, phi) 从世界坐标系，旋转到当前坐标系的极角（手机头与地磁的夹角），
    旋转到当前坐标系的滚角（手机头与地平面的夹角）
    :param walk_direction_bias: 手动偏移
    :param magic: pace_inference神奇公式需要的参数
    :param pace_inference:步幅推断器
    :param transform:
    :return:一个预测器
    """
    if magic is None:
        magic = [Magic_A, Magic_B, Magic_C]

    def predict(locus: PedestrianLocus):
        theta, phi = attitude if attitude else measure_initial_attitude(locus, 30)
        # 这里的姿态矩阵定义是：R^{EARTH}_{IMU}，因此p^{EARTH} = R^{EARTH}_{IMU} p^{IMU}
        imu_to_earth = Rotation.from_euler("ZYX", [theta, 0, phi])
        # imu_to_earth = measure_initial_attitude_advanced(locus, 30)

        # 提取传感器信息
        gyroscope_imu_frame, magnetometer_imu_frame, acceleration_imu_frame = locus.data["Gyroscope"][:, 1:], \
                                                                              locus.data["Magnetometer"][:, 1:], \
                                                                              locus.data["Linear Acceleration"][:, 1:]
        time_frame = locus.data["Gyroscope"][:, 0]

        # 记录力学信息
        info = __record_movement(locus, imu_to_earth, gyroscope_imu_frame,
                                 magnetometer_imu_frame, acceleration_imu_frame, time_frame,
                                 walk_direction_bias, euler=euler)

        info["magic"] = magic
        info["bias"] = {
            'test1':  # [ 1.2446017418969524 ,  0.3567613010836765 ,  -0.002618388787419153 ,  0.05225300414460947 ],
            # [ 1.3220182439646506 ,  0.3731370122755645 ,  -0.002788113290323491 ,  0.03915316431412718 ],#换euler
            # [ 3.1381789337959263 ,  0.2828957982662374 ,  -0.0019165129703986549 ,  -0.021079233864013003 ],
                [3.140000032806056, 0.37000133393000373, -0.0005319197831594395, 0.16380365553201406],  # 成功
            'test2':  # [ 1.680334985534297 ,  0.10824123684131158 ,  -0.007596952968952523 ,  -1.1648109992479414 ],
            # [ 1.839714508023309 ,  0.3687609318522305 ,  -3.346131131717134e-05 ,  0.15569048879227212 ],#700
                [1.6468032669834176, 0.3647953893513336, -0.0003272740880353399, 0.12973875466808252],  # 成功

            'test3': [0.0681082114306805, 0.1014283955679003, -0.0037350530192257216, -0.7657642038598802],  # 成功
            # 'test4': [ 3 ,  0.4943875462957725 ,  -0.030123552013203034 ,  -6.794281849340825 ],
            'test5':  # [ -2.3 ,  0.4642259602247386 ,  -0.0062636714640778674 ,  -1.391308642698473 ],
            # [ -2.8090144921172153 ,  0.34796575633627896 ,  -0.00027789144602138026 ,  0.11601484130512685 ],
            # [ -2.7534544399014544 ,  0.36366579896377044 ,  -0.00010002891279575242 ,  0.15150619947375188 ],
                [-2.5878134196058094, 0.36663420990895185, 2.8373284692811962e-05, 0.15726746525140284],  # 成功

            'test6':  # [ -0.11488087290831708 ,  0.36846546288830645 ,  5.9447627996637794e-05 ,  0.15989680171856857 ],#300
            # [ -0.1099549723362013 ,  0.31595650531190655 ,  -0.00014349823804356298 ,  0.12612606080868619 ],#300
                [-0.08010278941614628, 0.37046758661555784, 0.0001393571468202795, 0.16741268953637223],  # 280#成功
            # [ -0.004933180439836241 ,  0.3700288073008082 ,  0.0001481056465783818 ,  0.16402251464015316 ],
            'test7': [-0.46918666294603983, 2.487404071534488, 0.04179098671703162, 6.513396041090186],  # 成功
            'test8': [0.11961377772316462, 0.3694707544354912, -8.348216938939907e-05, 0.1610173376714257],
            # 成功
            'test9': [0.32683416459432657, 0.30712775535639186, -0.00289204322924796, -0.03692321568652859],
            'test10':  # [ 0.23467333213655409 ,  0.356059200622363 ,  -0.006047570578255694 ,  0.1476842431896482 ],
            # [ 0.259685232392236 ,  0.3603619620839632 ,  -0.006441288079198304 ,  0.11072776942196043 ],
                [0.2782529068739034, 0.3672676263067806, -0.006630434607615831, 0.05349965498428349],  # 成功，不翻转
            # [ 3.5879956081824824 ,  0.38035682075932137 ,  -0.006623410441072413 ,  0.17090560588116685 ],
            # [ 3.58526184049176 ,  0.4426402065997671 ,  -0.005941176258691512 ,  0.22683779874588753 ],
            # [ 3.5642753559363007 ,  0.5745068994801036 ,  -0.005500207948807497 ,  0.3151409825520777 ]
            'test11':  # [ 1.75 ,  0.37 ,  0.000155 ,  0.1638 ],
            # [ 1.75 ,  0.3340591887101659 ,  -0.00471874533409629 ,  -0.7749846638149595 ],
            # [ 1.75 ,  0.29526719723787354 ,  -0.004380162050346857 ,  -0.7125760592539472 ],
                [1.75, 0.32566865418627233, -0.0022041932951444004, -0.37013506076815267],  # 成功
        }
        info['args'] = {
            'test1': ("ZYX", None),
            'test2': ("ZYX", "transform_x"),
            'test3': ("ZXY", None),
            'test5': ("ZYX", None),
            'test6': ("ZYX", None),
            'test7': ("ZXY", None),
            'test8': ("ZXY", None),
            'test9': ("ZXY", None),
            'test10': ("ZXY", None),
            'test11': ("ZYX", None),

        }
        print('magic:[', walk_direction_bias, ", ", magic[0], ", ", magic[1], ", ", magic[2], "]")
        # 行人路经预推演
        info["inference_times"] = 0
        inference = pace_inference(info) if pace_inference else lambda x, y: PACE_STEP
        walk_positions, walk_directions = __simulated_walk(locus, info, inference, walk_direction_bias)
        info["gps_positions_temp"], info["gps_directions_temp"] = __aligned_with_gps(locus, info, walk_positions,
                                                                                     walk_directions)
        # if info["gps_positions_temp"] > 0:
        #     info["gps_positions_temp"] -= info["gps_positions_temp"][0]

        # 行人路经最终推演
        info["inference_times"] = 1
        inference = pace_inference(info) if pace_inference else lambda x, y: PACE_STEP
        walk_positions, walk_directions = __simulated_walk(locus, info, inference, walk_direction_bias)
        # if info["gps_positions_temp"] > 0:
        #     info["gps_positions_temp"] -= (info["gps_positions_temp"][locus.latest_gps_index])

        # 插值
        r = __aligned_with_gps(locus, info, walk_positions, walk_directions)
        if transform is not None:
            if transform == "transform_x":
                r = transform_x(locus, r)
        return r, info

    return predict


def __record_movement(locus, imu_to_earth, gyroscope_imu_frame,
                      magnetometer_imu_frame, acceleration_imu_frame, time_frame,
                      walk_direction_bias, euler="ZXY"):
    p, v = np.zeros(3), np.zeros(3)  # 获取一个初态

    thetas, phis, alphas, directions = [np.empty(len(time_frame) - 2) for _ in range(4)]
    speeds, accelerations, y_directions = [np.empty((len(time_frame) - 2, 3)) for _ in range(3)]
    for index, (gyroscope_imu, acceleration_imu, magnetometer_imu) in enumerate(
            zip(gyroscope_imu_frame[1: -1], acceleration_imu_frame[1: -1], magnetometer_imu_frame[1: -1])):
        delta_t = (time_frame[index + 2] - time_frame[index]) / 2
        # 姿态变化
        imu_to_earth = imu_to_earth * Rotation.from_euler("xyz", delta_t * gyroscope_imu)

        # 计算姿态角
        if euler == "ZYX":
            thetas[index], phis[index], alphas[index] = imu_to_earth.as_euler("ZYX")
        # 可以解决摆手机问题
        if euler == "ZXY":
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
            "walk_time": time_frame[1 + peaks_index], "locus": locus, "walk_direction_bias": walk_direction_bias}

    return info


def __simulated_walk(locus, info, inference, walk_direction_bias):
    peaks_index = info["peaks"]
    directions = info["directions"]

    walk_positions = np.zeros((len(peaks_index) + 1, 2))
    walk_directions = np.zeros(len(peaks_index))
    p = np.zeros(2)
    for index, peak in enumerate(peaks_index):
        direction = directions[peak - 2: peak + 3].mean()
        walk_directions[index] = direction + walk_direction_bias
        pace = inference(index, peak)
        p += pace * np.asarray([cos(np.pi / 2 + walk_directions[index]), sin(np.pi / 2 + walk_directions[index])])
        # y_direction = y_directions[peak][:2]
        # p += pace * y_direction / np.sqrt(y_direction[0] ** 2 + y_direction[1] ** 2)
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
                             walk_positions, kind='cubic', axis=0, fill_value="extrapolate") \
            (locus.y_frame["location_time"])
        directions = interp1d(walk_time, walk_directions, kind='cubic', axis=0, fill_value="extrapolate") \
            (locus.y_frame["location_time"])
    else:
        positions = None
        directions = None

    return positions, directions


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"])
    locus_predictor = locus_predictor()
    locus_predictor(dataset["图书馆2"])
