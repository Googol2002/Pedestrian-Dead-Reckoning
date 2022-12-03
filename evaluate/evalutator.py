import math
import os
from math import sqrt, degrees

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from numpy import arctan2, pi

from evaluate.test import eval_model
from locus_predictor.mature_locus_predictor import locus_predictor
from pace_predictor.predict_pace import magic_pace_inference, run_magic
from pedestrian_data import PedestrianLocus, PedestrianDataset, default_low_pass_filter

from geopy.distance import geodesic
import geopy.distance


def __bearing(x, y):
    return math.degrees(arctan2(x, y))


def evaluate_model(locus: PedestrianLocus, num, fixed_magic, compare=True):
    # (positions, directions), properties = predictor(locus)
    positions, directions = run_magic(locus, fixed_magic=fixed_magic)
    positions = positions - positions[locus.latest_gps_index]
    origin = geopy.Point(*locus.latest_gps_data)
    bearings = np.rad2deg(pi/2 - (pi/2 + directions))
    bearings += (bearings < 0) * 360
    destinations = np.array([list(geopy.distance.geodesic(kilometers=sqrt(x**2 + y**2)/1000).
                             destination(origin, bearing=__bearing(x, y)))[:2]
                             for x, y, bearing in zip(positions[:, 0], positions[:, 1], bearings)])

    location_input = pd.read_csv(os.path.join(locus.path, "Location_input.csv"), encoding="utf-8", dtype='float64')
    location_input_dropped = location_input.dropna()

    output = pd.DataFrame(location_input)
    output.iloc[-len(destinations):, [1, 2, 5]] = np.concatenate((
        destinations, np.expand_dims(bearings, axis=1)), axis=1)
    output.iloc[location_input_dropped.index] = location_input_dropped

    output.to_csv(os.path.join(locus.path, "Location_test_case_{}.csv".format(num)), index=False)
    output.to_csv(os.path.join(locus.path, "Location_output.csv".format(num)), index=False)


    if compare:
        eval_model(locus.path)


def plot_model_output(locus: PedestrianLocus, name, num):
    output_frame = pd.read_csv(os.path.join(locus.path, "Location_test_case_{}.csv".format(num)))
    input_frame = pd.read_csv(os.path.join(locus.path, "Location.csv"))

    plt.plot(output_frame["Longitude (°)"], output_frame["Latitude (°)"], label="Output")
    plt.plot(input_frame["Longitude (°)"], input_frame["Latitude (°)"], label="Input", linestyle="dashed")
    plt.title(name)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = PedestrianDataset(["Test", "TestSet"], window_size=200,
                                acceleration_filter=default_low_pass_filter)

    num = 0
    evaluate_model(dataset["test{}".format(num)], num, True, compare=True)
    plot_model_output(dataset["test{}".format(num)], "test{}".format(num), num)