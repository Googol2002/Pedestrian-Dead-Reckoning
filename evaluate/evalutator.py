import os
from math import sqrt, degrees

import numpy as np
import pandas as pd
from numpy import arctan2, pi

from evaluate.test import eval_model
from locus_predictor.mature_locus_predictor import locus_predictor
from pace_predictor.predict_pace import magic_pace_inference
from pedestrian_data import PedestrianLocus, PedestrianDataset, default_low_pass_filter

from geopy.distance import geodesic
import geopy.distance


def evaluate_model(locus: PedestrianLocus, predictor):
    (positions, directions), properties = predictor(locus)
    origin = geopy.Point(*locus.origin)
    bearings = np.rad2deg(pi/2 - (pi/2 + directions))
    destinations = np.array([list(geopy.distance.geodesic(kilometers=sqrt(x**2 + y**2)/1000).
                             destination(origin, bearing=np.degrees(arctan2(x, y))))[:2]
                             for x, y, bearing in zip(positions[:, 0], positions[:, 1], bearings)])

    location_input = pd.read_csv(os.path.join(locus.path, "Location_input.csv"), encoding="utf-8", dtype='float64')
    location_input_dropped = location_input.dropna()

    output = pd.DataFrame(location_input)
    output.iloc[-len(destinations):, [1, 2, 5]] = np.concatenate((
        destinations, np.expand_dims(bearings, axis=1)), axis=1)
    output.iloc[location_input_dropped.index] = location_input_dropped

    output.to_csv(os.path.join(locus.path, "Location_output.csv"), index=False)

    eval_model(locus.path)


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer", "Hand-Walk"], window_size=200,
                                acceleration_filter=default_low_pass_filter)

    evaluate_model(dataset["test_case0"], predictor=locus_predictor(pace_inference=magic_pace_inference))