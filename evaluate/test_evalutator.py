""" 用于我们收集数据的测试集的评估函数 """
import math
import os
from math import sqrt, degrees
import pmdarima as pm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from numpy import arctan2, pi

from evaluate.test import eval_model
from locus_predictor.mature_locus_predictor import locus_predictor
from pedestrian_data import PedestrianLocus, PedestrianDataset, default_low_pass_filter
import statsmodels.api as sm
from geopy.distance import geodesic
import geopy.distance


def __bearing(x, y):
    return math.degrees(arctan2(x, y))


def evaluate_model(locus: PedestrianLocus, pace_inference, compare=True):
    predictor = locus_predictor(pace_inference=pace_inference)
    (positions, directions), properties = predictor(locus)
    positions = positions - positions[locus.latest_gps_index]
    origin = geopy.Point(*locus.latest_gps_data)
    bearings = np.rad2deg(pi / 2 - (pi / 2 + directions))
    bearings += (bearings < 0) * 360
    bearings %= 360
    # 计算出 bias 再来一次
    gt = locus.y_frame.iloc[:, 5].dropna()
    walk_direction_bias = ema_cal_walk_direction_bias(bearings[:len(gt)], gt)

    predictor = locus_predictor(pace_inference=pace_inference, walk_direction_bias=walk_direction_bias)
    (positions, directions), properties = predictor(locus)
    positions = positions - positions[locus.latest_gps_index]
    origin = geopy.Point(*locus.latest_gps_data)
    bearings = np.rad2deg(pi / 2 - (pi / 2 + directions))
    bearings += (bearings < 0) * 360
    bearings %= 360

    # ARIMA 建模误差项 扔掉前8个
    # gt = locus.y_frame.iloc[:, 5].dropna()
    # head_10_error = gt[2:] - bearings[2:len(gt)]
    # ARIMA = pm.auto_arima(head_10_error).fit(head_10_error)
    # lag = ARIMA.predict(len(bearings) - len(gt))  # 补偿项
    # # ARIMA = sm.tsa.arima.ARIMA(head_10_error, order=(2, 1, 1)).fit()
    # # lag = ARIMA.forecast(len(bearings) - len(gt))  # 补偿项
    # bearings[len(gt):] += lag
    # bearings += (bearings < 0) * 360
    # bearings %= 360

    # ARIMA 建模误差项 扔掉前8个
    # gt1 = locus.y_frame.iloc[:, 0].dropna()
    # gt2 = locus.y_frame.iloc[:, 1].dropna()
    # head_10_error_1 = gt1[5:] - positions[5:len(gt1), 0]
    # head_10_error_2 = gt2[5:] - positions[5:len(gt2), 1]
    # ARIMA_1 = pm.auto_arima(head_10_error_1).fit(head_10_error_1)
    # ARIMA_2 = pm.auto_arima(head_10_error_2).fit(head_10_error_2)
    # lag_1 = ARIMA_1.predict(len(positions) - len(gt))  # 补偿项
    # lag_2 = ARIMA_2.predict(len(positions) - len(gt))  # 补偿项
    # # ARIMA = sm.tsa.arima.ARIMA(head_10_error, order=(2, 1, 1)).fit()
    # # lag = ARIMA.forecast(len(bearings) - len(gt))  # 补偿项
    # positions[len(gt):, 0] += lag_1
    # positions[len(gt):, 1] += lag_2
    # bearings += (bearings < 0) * 360
    # bearings %= 360

    destinations = np.array([list(geopy.distance.geodesic(kilometers=sqrt(x ** 2 + y ** 2) / 1000).
                                  destination(origin, bearing=__bearing(x, y)))[:2]
                             for x, y, bearing in zip(positions[:, 0], positions[:, 1], bearings)])

    location_input = pd.read_csv(os.path.join(locus.path, "Location_input.csv"), encoding="utf-8", dtype='float64')
    location_input_dropped = location_input.dropna()

    output = pd.DataFrame(location_input)
    output.iloc[-len(destinations):, [1, 2, 5]] = np.concatenate((
        destinations, np.expand_dims(bearings, axis=1)), axis=1)
    output.iloc[location_input_dropped.index] = location_input_dropped

    output.to_csv(os.path.join(locus.path, "Location_output.csv"), index=False)

    if compare:
        (dist_error, dir_error) = eval_model(locus.path)
        return dist_error, dir_error


def plot_model_output(locus: PedestrianLocus):
    output_frame = pd.read_csv(os.path.join(locus.path, "Location_output.csv"))
    input_frame = pd.read_csv(os.path.join(locus.path, "Location_input.csv"))

    plt.plot(output_frame["Longitude (°)"], output_frame["Latitude (°)"], label="Output")
    plt.plot(input_frame["Longitude (°)"], input_frame["Latitude (°)"], label="Input")
    plt.legend()
    plt.show()


def ema_cal_walk_direction_bias(bearings, gt, decay=0.6, length=5):
    """ EMA 计算 walk_direction_bias """
    data = bearings - gt
    res = 0
    length = min(length, len(bearings))
    for idx in range(length):
        res = res + pow(decay, length - 1 - idx) * data[len(data) - length + idx]
    c = (1 - decay) / (1 - pow(decay, length))
    res *= c
    bias = res / 180 * math.pi
    return bias


if __name__ == "__main__":
    pass
