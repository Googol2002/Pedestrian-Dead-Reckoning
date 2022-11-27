import os

import numpy as np
import pandas as pd

from evaluate.test import eval_model
from locus_predictor.mature_locus_predictor import predict
from pedestrian_data import PedestrianLocus, PedestrianDataset, default_low_pass_filter


def evaluate_model(locus: PedestrianLocus, predictor):
    (positions, directions), properties = predictor(locus)

    original_y_frame = pd.DataFrame(locus.y_frame.dropna())

    output = pd.DataFrame(locus.y_frame)
    output.iloc[len(original_y_frame):, [1, 2, 5]] = np.concatenate((
        positions, np.expand_dims(directions, axis=1)))
    output[original_y_frame.index] = original_y_frame

    output.to_csv(os.path.join(locus.path, "Location_output.csv"))

    eval_model(locus.path)


if __name__ == "__main__":
    dataset = PedestrianDataset(["Magnetometer", "Hand-Walk"], window_size=200,
                                acceleration_filter=default_low_pass_filter)

    evaluate_model(dataset["test_case0"], predictor=predict)