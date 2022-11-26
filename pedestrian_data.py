import math
import os
import json
import codecs
import scipy.signal as signal
from collections.abc import Iterable
from functools import reduce

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from geographiclib.geodesic import Geodesic

with codecs.open(r"config/config.json", 'r', 'utf-8') as config_file:
    config_data = json.load(config_file)
    PEDESTRIAN_DATA_PATH = config_data["Data-Path"]

__SCENARIOS_FILTERED = {".git"}
_scenarios = {scenario: [sample for sample in os.listdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario))
                         if os.path.isdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario, sample))] for scenario in
              filter(lambda f: f not in __SCENARIOS_FILTERED, os.listdir(PEDESTRIAN_DATA_PATH))
              if os.path.isdir(os.path.join(PEDESTRIAN_DATA_PATH, scenario))}


def default_low_pass_filter(data):
    # 采样频率为50 hz, 要滤除10hz以上频率成分，
    # 即截至频率为7.5hz, 则wn = 2 * 10 / 50 = 0.4 == > Wn = 0.4
    b, a = signal.butter(8, 0.4, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    columns = data.columns[1:]

    for ax in columns:
        data[ax] = signal.filtfilt(b, a, data[ax])  # data为要过滤的信号


# 效果不好
# def default_high_pass_filter(data):
#     # 采样频率为50 hz, 要滤除0.5hz以下频率成分，
#     # 即截至频率为0.5hz, 则wn = 2 * 0.5 / 50 = 0.02 == > Wn = 0.02
#     b, a = signal.butter(8, 0.02, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
#     columns = data.columns[1:]
#
#     for ax in columns:
#         data[ax] = signal.filtfilt(b, a, data[ax])  # data为要过滤的信号


class PedestrianDataset(Iterable):

    def __init__(self, scenarios: list, window_size=200,
                 acceleration_filter=default_low_pass_filter):
        self.loci = dict()

        for paths in (zip(_scenarios[s],
                          [os.path.join(PEDESTRIAN_DATA_PATH, s, f) for f in _scenarios[s]])
                      for s in scenarios):
            for k, path in paths:
                self.loci[k] = PedestrianLocus(path, window_size,
                                               acceleration_filter=acceleration_filter)

    def __len__(self):
        return len(self.loci)

    def __getitem__(self, index):
        return self.loci[index]

    def __iter__(self):
        return ((k, v) for k, v in self.loci.items())


class PedestrianLocus(Dataset):

    def __init__(self, path, window_size,
                 acceleration_filter=None, gyroscope_filter=None):
        # 第一个的时间戳将作为最终的时间戳
        x_sub_frame_names = [("Accelerometer", "Accelerometer"),
                             ("Gyroscope", "Gyroscope"),
                             ("Magnetometer", "Magnetometer"),
                             ("Linear Acceleration", "Linear Acceleration"),
                             ("Linear Acceleration", "Linear Accelerometer")]

        x_sub_frames = {frame_name: pd.read_csv(os.path.join(path, file_name) + ".csv", encoding="utf-8",
                                                dtype=np.float64) for frame_name, file_name
                        in x_sub_frame_names if os.path.exists(os.path.join(path, file_name) + ".csv")}

        # 某些同学手机传感器数据文件列名有重名现象，适配了这一情况
        for frame_name, frame in x_sub_frames.items():
            frame.columns = map(
                lambda col_name: "{}.{}".format(frame_name, col_name) if col_name != "Time (s)" else "Time (s)",
                frame.columns)

        self.y_frame = pd.read_csv(os.path.join(path, "Location.csv"), encoding="utf-8")
        self.window_size = window_size

        # 预处理阶段
        if acceleration_filter is not None:
            acceleration_filter(x_sub_frames["Accelerometer"])
            acceleration_filter(x_sub_frames["Linear Acceleration"])
        if gyroscope_filter is not None:
            gyroscope_filter(x_sub_frames["Gyroscope"])

        # 前几个数据点有噪声啊
        self.relative_location = self.y_frame[["Time (s)", "Latitude (°)", "Longitude (°)"]].dropna()
        origin_latitude, origin_longitude = np.mean(self.relative_location["Latitude (°)"][4:8]), \
                                            np.mean(self.relative_location["Longitude (°)"][4:8])

        if not math.isnan(origin_latitude) and not math.isnan(origin_latitude):
            geo_infos = [Geodesic.WGS84.Inverse(origin_latitude, origin_longitude,
                                                self.relative_location["Latitude (°)"][i],
                                                self.relative_location["Longitude (°)"][i])
                         for i in range(len(self.relative_location))]
            self.relative_location["relative_x (m)"] = [geo_info["s12"] * math.cos(
                math.pi / 2 - math.radians(geo_info["azi1"])) for geo_info in geo_infos]
            self.relative_location["relative_y (m)"] = [geo_info["s12"] * math.sin(
                math.pi / 2 - math.radians(geo_info["azi1"])) for geo_info in geo_infos]

        # 合并表
        self.x_frame = reduce(lambda left, right: pd.merge_asof(left, right, on="Time (s)", direction="nearest"),
                              x_sub_frames.values())

        time_table = pd.DataFrame({"Time (s)": self.x_frame["Time (s)"],
                                   "nearest_time": self.x_frame["Time (s)"]})
        self.y_frame = pd.merge_asof(self.y_frame, time_table, on="Time (s)", direction="nearest")
        self.y_frame.rename(columns={"Time (s)": "location_time"}, inplace=True)

        self.frame = pd.merge(self.x_frame, self.y_frame, how="left",
                              left_on="Time (s)", right_on="nearest_time")

        self.data_columns = {k: sub_frame.columns for k, sub_frame in x_sub_frames.items()}
        self.data = {k: self.frame[sub_frame.columns].to_numpy(dtype=float)
                     for k, sub_frame in x_sub_frames.items()}
        self.data["Location"] = self.frame[self.y_frame.columns].to_numpy(dtype=float)
        self.data_columns["Location"] = self.y_frame.columns

    def __getitem__(self, index):
        return {k: v[index: index + self.window_size] for k, v in self.data.items()}

    def __len__(self):
        return len(self.frame) - self.window_size + 1

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def columns_info(self):
        return self.data_columns


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer"], window_size=200,
                                acceleration_filter=default_low_pass_filter)

    for name, locus in dataset:
        print("正在遍历移动轨迹{}... \n".format(name))

        for sample in locus:
            for k, v in sample.items():
                print(k + ":" + str(v.shape))
            break

        print(locus.columns_info())
        break
