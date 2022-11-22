# -*- coding: UTF-8 -*- #
"""
@filename:plot_locus.py
@author:201300086
@time:2022-11-21
"""
from pedestrian_data import PedestrianDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

test_range = (0, 5)  # 画图范围
CUT_BEGIN = 10  # 删掉前几秒的数据，因为GPS不准
l, r = test_range

dataset = PedestrianDataset(["test_case0"], window_size=100)  # 指定文件夹
for num, (name, locus) in enumerate(dataset):
    if num in range(l, r):
        print("正在遍历移动轨迹{}... \n".format(name))
        locus_pair = np.array(list(zip(locus.y_frame["Latitude (°)"], locus.y_frame["Longitude (°)"])))
        print("轨迹长度: ", len(locus_pair))
        Lati = locus_pair.T[0][CUT_BEGIN:]
        Longi = locus_pair.T[1][CUT_BEGIN:]
        plt.xlabel("Latitude (°)")
        plt.ylabel("Longitude (°)")
        plt.text(Lati[0], Longi[0], 's', fontsize=10)
        plt.text(Lati[-1], Longi[-1], 'e', fontsize=10)
        plt.plot(Lati, Longi, '+', markersize=1, label="{}".format(name))
        plt.legend(loc="upper right")

    if num >= r:
        break
plt.show()
