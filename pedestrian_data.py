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


def default_mask():
    def mask(y_frame: pd.DataFrame):
        quantile_10 = len(y_frame) - len(y_frame) * 9 // 10
        y_frame.iloc[quantile_10:, [1, 2, 5]] = np.nan

        return y_frame

    return mask

def do_not_mask():
    def mask(y_frame: pd.DataFrame):
        return y_frame

    return mask

class PedestrianDataset(Iterable):

    def __init__(self, scenarios: list, window_size=200, mask=None, skip_len=0,
                 acceleration_filter=default_low_pass_filter):
        self.loci = dict()
        mask = mask if mask else default_mask()

        for paths in (zip(_scenarios[s],
                          [os.path.join(PEDESTRIAN_DATA_PATH, s, f) for f in _scenarios[s]])
                      for s in scenarios):
            for k, path in paths:
                self.loci[k] = PedestrianLocus(path, window_size, mask, skip_len,
                                               acceleration_filter=acceleration_filter)

    def __len__(self):
        return len(self.loci)

    def __getitem__(self, index):
        return self.loci[index]

    def __iter__(self):
        return ((k, v) for k, v in self.loci.items())


class PedestrianLocus(Dataset):

    def __init__(self, path, window_size, mask, skip_len,
                 acceleration_filter=default_low_pass_filter,
                 gyroscope_filter=None):
        self.path = path
        self.window_size = window_size

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

        # 读取Location_input.csv或Mask Location.csv
        if os.path.exists(os.path.join(path, "Location_input.csv")):
            self.y_frame = pd.read_csv(os.path.join(path, "Location_input.csv"), encoding="utf-8", dtype='float64')
        else:
            self.y_frame = mask(
                pd.read_csv(os.path.join(path, "Location.csv"), encoding="utf-8", dtype='float64'))
            self.y_frame.to_csv(os.path.join(path, "Location_input.csv"), index=False)

        self.y_frame = self.y_frame.iloc[skip_len:].reset_index(drop=True)
        if os.path.exists(os.path.join(path, "Location.csv")):
            self.ans = pd.read_csv(os.path.join(path, "Location.csv"), encoding="utf-8", dtype='float64')
            self.ans_relative_location, _ = self.__process_gps_data(self.ans, "relative_x (m)", "relative_y (m)")
        # 分别加工GPS数据
        self.relative_location, self.origin = self.__process_gps_data(self.y_frame, "relative_x (m)", "relative_y (m)")
        # 利用最终的GPS数据矫正
        self.latest_gps_index = len(self.y_frame.dropna()) - 1
        self.latest_gps_data = (self.y_frame["Latitude (°)"][self.latest_gps_index],
                                self.y_frame["Longitude (°)"][self.latest_gps_index])

        # 预处理阶段
        if acceleration_filter is not None:
            acceleration_filter(x_sub_frames["Accelerometer"])
            acceleration_filter(x_sub_frames["Linear Acceleration"])
        if gyroscope_filter is not None:
            gyroscope_filter(x_sub_frames["Gyroscope"])

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

    @staticmethod
    def __process_gps_data(source_frame, x_label, y_label):
        # 前几个数据点有噪声啊
        relative_location = source_frame[["Time (s)", "Latitude (°)", "Longitude (°)"]].dropna()
        origin_latitude, origin_longitude = np.mean(relative_location["Latitude (°)"][:1]), \
                                            np.mean(relative_location["Longitude (°)"][:1])

        origin = (origin_latitude, origin_longitude)
        if not math.isnan(origin_latitude) and not math.isnan(origin_latitude):
            geo_infos = [Geodesic.WGS84.Inverse(origin_latitude, origin_longitude,
                                                relative_location["Latitude (°)"][i],
                                                relative_location["Longitude (°)"][i])
                         for i in range(len(relative_location))]
            relative_location[x_label] = [geo_info["s12"] * math.cos(
                math.pi / 2 - math.radians(geo_info["azi1"])) for geo_info in geo_infos]
            relative_location[y_label] = [geo_info["s12"] * math.sin(
                math.pi / 2 - math.radians(geo_info["azi1"])) for geo_info in geo_infos]

        return relative_location, origin

    def columns_info(self):
        return self.data_columns


if __name__ == "__main__":
    dataset = PedestrianDataset(["Bag-Ride", "Bag-Walk"], window_size=200,
                                acceleration_filter=default_low_pass_filter)

    for name, locus in dataset:
        print("正在遍历移动轨迹{}... \n".format(name))

        for sample in locus:
            for k, v in sample.items():
                print(k + ":" + str(v.shape))
            break

        print(locus.columns_info())
        break

