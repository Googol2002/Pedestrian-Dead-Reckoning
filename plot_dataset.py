# -*- coding: UTF-8 -*- #
"""
@filename:plot_dataset.py
@author:201300086
@time:2022-11-21
"""
from pedestrian_data import PedestrianDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# 步行轨迹图
# l, r = (0, 5)  # 画图范围
# CUT_BEGIN = 10  # 删掉前几秒的数据，因为GPS不准
#
# dataset = PedestrianDataset(["test_case0"], window_size=100)  # 指定文件夹
# for num, (name, locus) in enumerate(dataset):
#     if num in range(l, r):
#         print("正在遍历移动轨迹{}... \n".format(name))
#         locus_pair = np.array(list(zip(locus.y_frame["Latitude (°)"], locus.y_frame["Longitude (°)"])))
#         print("轨迹长度: ", len(locus_pair))
#         Lati = locus_pair.T[0][CUT_BEGIN:]
#         Longi = locus_pair.T[1][CUT_BEGIN:]
#         plt.xlabel("Latitude (°)")
#         plt.ylabel("Longitude (°)")
#         plt.text(Lati[0], Longi[0], 's', fontsize=10)
#         plt.text(Lati[-1], Longi[-1], 'e', fontsize=10)
#         plt.plot(Lati, Longi, '+', markersize=1, label="{}".format(name))
#         plt.legend(loc="upper right")
#     if num >= r:
#         break
# plt.show()




# 重力加速度变化图
l, r = (10, 14)  # 画图范围
dataset = PedestrianDataset(["Hand-Walk"], window_size=10000)

for num, (name, locus) in enumerate(dataset):
    if num in range(l, r):
        print("正在遍历移动轨迹{}... \n".format(name))
        plt.subplot(22 * 10 + num - 10 + 1)
        for sample in locus:
            print(len(locus))
            # print(sample.keys())
            a = sample['Accelerometer']
            b = sample['Linear Acceleration']
            minus = (a - b)
            c = np.array(list(map(lambda x: np.linalg.norm(x), minus)))
            print(c.mean())
            plt.plot(range(len(c)), c)
            plt.title(label="{}".format(name))
            # plt.legend(loc="upper right")
            break
    if num >= r:
        break
plt.show()
