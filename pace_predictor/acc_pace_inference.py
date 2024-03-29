import math
from scipy.signal import find_peaks
import numpy as np

MIN_PERIOD = 20
PROMINENCE = (0.05, None)
Magic_A=0.37
Magic_B=0.000155
Magic_C=0.1638
def pace_inference(info):
    accelerations = info["accelerations"]
    a_vec = np.sqrt(np.power(accelerations[:, 0], 2) + np.power(accelerations[:, 1], 2))
    time = info["time"]
    peaks_index, _ = find_peaks(a_vec, distance=MIN_PERIOD, prominence=PROMINENCE)
    walk_time = time[peaks_index]
    steplength = []
    magic_a,magic_b,magic_c=Magic_A,Magic_B,Magic_C
    for i in range(len(walk_time) - 1):
        W = (walk_time[i] if i == 0 else walk_time[i + 1] - walk_time[i]) * 1000
        peak = (a_vec[peaks_index[i + 1]] + a_vec[peaks_index[i]]) / 2
        valley = np.min(a_vec[peaks_index[i]:peaks_index[i + 1]])
        H = peak - valley
        magic_num=magic_a - magic_b * W + magic_c * math.sqrt(H)
        steplength.append(magic_num)
    steplength = np.array(steplength)#.clip(0.6, 0.75)
   # print("pace predict mean:",steplength.mean())

    def inference(index, peak):
        a_earth = accelerations[peak]
        # index 从0 到 步数
        # peak 从0 到 总数据数（可以直接用来访问info中的数据
        return steplength[index] if index < len(steplength) else ema(np.asarray(steplength))
        # return steplength[index] if index < len(steplength) else np.asarray(steplength).mean()

    return inference


def ema(data: np.ndarray, decay=0.9):
    """ EMA平滑预测 """
    res = 0
    L = len(data)
    for idx in range(len(data)):
        res = res + pow(decay, L - 1 - idx) * data[idx]
    c = (1 - decay) / (1 - pow(decay, L))
    res *= c
    return res
