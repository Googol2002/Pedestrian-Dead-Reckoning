import math

from scipy.signal import find_peaks
import numpy as np

MIN_PERIOD = 20
PROMINENCE = (0.05, None)


def pace_inference(info):
    accelerations = info["accelerations"]
    walk_time = info["walk_time"]
    ay = accelerations[:, 2]
    peaks_index, _ = find_peaks(ay, distance=MIN_PERIOD, prominence=PROMINENCE)

    steplength = []
    for i in range(len(walk_time) - 1):
        W = (walk_time[i] if i == 0 else walk_time[i + 1] - walk_time[i]) * 1000
        peak = (ay[peaks_index[i + 1]] + ay[peaks_index[i]]) / 2
        valley = np.min(ay[peaks_index[i]:peaks_index[i + 1]])
        H = peak - valley

        steplength.append(0.37 - 0.000155 * W + 0.1638 * math.sqrt(H))

    def inference(index, peak):
        a_earth = accelerations[peak]
        # index 从0 到 步数
        # peak 从0 到 总数据数（可以直接用来访问info中的数据
        return steplength[index] if index < len(steplength) else np.asarray(steplength).mean()

    return inference
