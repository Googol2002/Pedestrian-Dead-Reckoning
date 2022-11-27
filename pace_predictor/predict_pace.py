# -*- coding: UTF-8 -*- #
"""
@filename:predict_pace.py
@author:201300086
@time:2022-11-26
"""
from pedestrian_data import PedestrianDataset,do_not_mask
from geopy.distance import geodesic
import math
from scipy.signal import find_peaks
from evaluate.test import get_dist_error_meters
import numpy as np
import pandas as pd
import os
MIN_PERIOD = 20
PROMINENCE = (0.05, None)
import matplotlib
import matplotlib.pyplot as plt
from locus_predictor.mature_locus_predictor import locus_predictor,__simulated_walk,__aligned_with_gps
from plot_dataset import plot_locus
from acc_pace_inference import pace_inference
matplotlib.use('TkAgg')

Train_rate=0.1#使用前10%标记预测步幅

def baseline_pace_inference(info):
    locus=info['locus']
    transform = 1

    if (info["inference_times"] == 1):
        positions = info["gps_positions_temp"]
        positions=positions-positions[0]

        positions=positions.T
        start_to_end_positions = np.linalg.norm(np.array([positions[0][-1], positions[1][-1]]) - np.array([positions[0][0], positions[1][0]]))
        Lati = locus.relative_location["relative_x (m)"].to_numpy()
        Longi = locus.relative_location["relative_y (m)"].to_numpy()
        start_to_end_GPS = np.linalg.norm(np.array([Lati[-1], Longi[-1]]) - np.array([Lati[0], Longi[0]]))
        #print("Compare before correction:",start_to_end_positions,start_to_end_GPS)
        transform = start_to_end_GPS/start_to_end_positions#pace整体乘一个比例
        print(transform)


    #总位移除步数
    walk_direction_bias=info['walk_direction_bias']
    Lati = locus.relative_location["relative_x (m)"].to_numpy()
    Longi = locus.relative_location["relative_y (m)"].to_numpy()
    start_to_end_GPS= np.linalg.norm(np.array([Lati[-1],Longi[-1]]) - np.array([Lati[0],Longi[0]]))

    dist_list=[]
    dist_list_whole=[]
    normal_step=0
    for i in range(0,int(len(Lati) * Train_rate)):
        vec1=np.array([Lati[i],Longi[i]])
        vec2=np.array([Lati[i+1],Longi[i+1]])
        dist = np.linalg.norm(vec1 - vec2)#欧氏距离
        #print(dist)
        dist_list_whole.append(dist)
        if dist<0.75:#dist>0.6 and
            dist_list.append(dist)
            normal_step+=1
    dist_list=np.array(dist_list)

    def inference(index, peak):
        pace=dist_list.sum()/normal_step
        #print("baseline pace=",pace,normal_step)
        return pace * transform

    return inference


def idiot_pace_inference(info):
    def inference(index, peak):
        return 0.8
    return inference

def save_output(position,location_time,output_path,locus):
    #print("len(position):",len(position))
    df = pd.DataFrame(position)
    df["Time (s)"] = location_time
    df = df.iloc[:, [2, 0, 1]]
    df.columns = ["Time (s)", "Latitude (°)", "Longitude (°)"]
    #df.to_csv(os.path.join(output_path, "Location_output.csv"), sep=',', index=False)
    # gt = pd.read_csv(os.path.join(output_path, "Location.csv"))
    # pred = pd.read_csv(os.path.join(output_path, "Location_output.csv"))
    #print(locus.relative_location)
    # gt=locus.relative_location.iloc[:, [0, 4, 3]]
    # gt['relative_y (m)']*=-1
    gt = locus.relative_location.iloc[:, [0, 3, 4]]
    #print(gt)
    pred =df
    dist_error = get_dist_error_meters(gt, pred)
    print("error：", dist_error)

def plot_result(data):
    path="C:\\Users\\Shawn\\Desktop\\python_work\\pytorch\\Dataset-of-Pedestrian-Dead-Reckoning\\Hand-Walk"
    #data="Hand-Walk-02-004"
    output_path=os.path.join(path, data)
    dataset = PedestrianDataset(["Hand-Walk"], window_size=1000,mask=do_not_mask(0))
    locus=dataset[data]
    location_time=locus.y_frame["location_time"]
    predictor_base = locus_predictor(pace_inference=baseline_pace_inference,walk_direction_bias=0.27)
    (position,direction),info=predictor_base(locus,)
    # print(info['walk_time'])

    save_output(position,location_time,output_path,locus)
    position=position.T

    Lati = locus.relative_location["relative_x (m)"].to_numpy()
    Longi = locus.relative_location["relative_y (m)"].to_numpy()
    print('GPS origin',Lati[0],Longi[0])
    print('position origin', (position[0][0],position[1][0]))
    plot_locus(position[0],position[1],label='baseline_pace')

    # predictor_idiot = locus_predictor(pace_inference=idiot_pace_inference)
    # (position,_),__=predictor_idiot(locus, )
    # position=position.T
    # plot_locus(position[0],position[1],label='pace=0.8')
    predictor_acc = locus_predictor(pace_inference=pace_inference,walk_direction_bias=0.27)
    (position,direction),info=predictor_acc(locus)
    # save_output(position,location_time,output_path,locus)
    position=position.T
    #print("acc_pace_len:",len(position[0]))
    plot_locus(position[0],position[1],label='acc_pace')


    Lati=locus.relative_location["relative_x (m)"].to_numpy()
    Longi=locus.relative_location["relative_y (m)"].to_numpy()
    #print("GPS:",len(Lati))
    #plot_locus(Longi*-1,Lati,label='GPS')
    #plot_locus(Longi , Lati* -1, label='GPS')

    plot_locus(Lati,Longi,label='GPS')
    plt.title(data)

plt.subplot(221)
plot_result("Hand-Walk-02-003")
plt.subplot(222)
plot_result("Hand-Walk-02-002")
plt.subplot(223)
plot_result("Hand-Walk-02-005")
plt.subplot(224)
plot_result("Hand-Walk-02-006")
#plot_result("Hand-Walk-02-004")
plt.show()