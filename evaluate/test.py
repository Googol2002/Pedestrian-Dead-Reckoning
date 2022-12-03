import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic


def eval_model(test_path):
    gt = pd.read_csv(os.path.join(test_path, "Location.csv"))
    pred = pd.read_csv(os.path.join(test_path, "Location_output.csv"))
    dist_error = get_dist_error(gt, pred)
    dir_error = get_dir_error(gt, pred)
    print("Distances error: ", dist_error)
    print("Direction error: ", dir_error)
    return (dist_error,dir_error)


def get_dir_error(gt, pred):
    dir_list = []
    cnt = 0
    for i in range(int(len(gt) * 0.1), len(gt)):
        dir = min(abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]), 360 - abs(gt[gt.columns[5]][i] - pred[pred.columns[5]][i]))
        dir_list.append(dir)
        if dir > 15:
            cnt += 1
    error = sum(dir_list) / len(dir_list)
    print(">15° percent: {}%".format(cnt/len(dir_list)*100))
    return error


def get_dist_error(gt, pred):
    print("local_error")
    dist_list = []
    for i in range(int(len(gt) * 0.1), len(gt)):
        dist = geodesic((gt[gt.columns[1]][i], gt[gt.columns[2]][i]), (pred[pred.columns[1]][i], pred[pred.columns[2]][i])).meters
        dist_list.append(dist)
    error = sum(dist_list) / len(dist_list)
    return error

def get_dist_error_meters(gt, pred):
    dist_list = []
    #print("len(gt)", len(gt))
    for i in range(int(len(gt) * 0.1), len(gt)):
        vec1=np.array([gt[gt.columns[1]][i], gt[gt.columns[2]][i]])
        vec2=np.array([pred[pred.columns[1]][i], pred[pred.columns[2]][i]])
        dist = np.linalg.norm(vec1 - vec2)
        #print(vec1,vec2,dist)
        dist_list.append(dist)
        #print(*dist_list)
    error = sum(dist_list) / len(dist_list)
    return error

def get_dist_train_error_meters(gt, pred):#训练时只能按前10%GPS更新参数，传入的就已经是10%
    dist_list = []
    #print("len(gt)", len(gt))
    for i in range(0,int(len(gt) )):
        vec1=np.array([gt[gt.columns[1]][i], gt[gt.columns[2]][i]])
        vec2=np.array([pred[pred.columns[1]][i], pred[pred.columns[2]][i]])
        dist = np.linalg.norm(vec1 - vec2)
        #print(vec1,vec2,dist)
        dist_list.append(dist)
        #print(*dist_list)
    error = sum(dist_list) / len(dist_list)
    return error

if __name__ == "__main__":
    eval_model("test_case0")
