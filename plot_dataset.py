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


# 步行轨迹图
CUT_BEGIN = 0  # 删掉前几秒的数据，因为GPS不准
def plot_locus(Lati,Longi,label="TBD",cut_begin=CUT_BEGIN):
    Lati = Lati[cut_begin:]
    Longi = Longi[cut_begin:]
    # plt.xlabel("Latitude (°)")
    # plt.ylabel("Longitude (°)")
    plt.text(Lati[0], Longi[0], 's', fontsize=10)
    plt.text(Lati[-1], Longi[-1], 'e', fontsize=10)
    plt.plot(Lati, Longi, label=label)#, '+', markersize=1
    plt.legend(loc=0)
# l, r = (0, 9)  # 画图范围
# dataset = PedestrianDataset(["Hand-Walk"], window_size=100)  # 指定文件夹
# for num, (name, locus) in enumerate(dataset):
#     if num in range(l, r):
#         print("正在遍历移动轨迹{}... \n".format(name))
#         locus_pair = np.array(list(zip(locus.y_frame["Latitude (°)"], locus.y_frame["Longitude (°)"])))
#         print("轨迹长度: ", len(locus_pair))
#         Lati = locus_pair.T[0]
#         Longi = locus_pair.T[1]
#         plot_locus(Lati,Longi, label="{}".format(name))
#     if num >= r:
#         break
#plt.show()



# 重力加速度变化图
def plot_gravity(sample:dict,title="TBD"):
    a = sample['Accelerometer']
    b = sample['Linear Acceleration']
    minus = (a - b)
    print(minus)
    c = np.array(list(map(lambda x: np.linalg.norm(x), minus)))
    print(c.mean())
    plt.plot(range(len(c)), c)
    plt.title(title)

# l, r = (0, 9)  # 画图范围
# dataset = PedestrianDataset(["Hand-Walk"], window_size=1000)
#
# for num, (name, locus) in enumerate(dataset):
#     if num in range(l, r):
#         print("正在遍历移动轨迹{}... \n".format(name))
#         plt.subplot(33 * 10 + num%10 + 1)
#         for sample in locus:
#             #print(len(locus))
#             plot_gravity(sample,title="{}".format(name))
#             break
#     if num >= r:
#         break
# plt.show()
if __name__ == "__main__":
    matplotlib.use('TkAgg')
    dataset = PedestrianDataset(["Hand-Walk"], window_size=1000)
    locus=dataset['Hand-Walk-02-001']
    time_frame = locus.data["Gyroscope"][:, 0]
#position,info=predict(locus,walk=True)
