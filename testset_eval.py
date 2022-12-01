"""
我们收集数据的测试集评估
 """
import numpy as np
from evaluate.test_evalutator import evaluate_model
from pace_predictor.predict_pace import magic_pace_inference
from pace_predictor.acc_pace_inference import pace_inference
from pedestrian_data import PedestrianDataset, default_low_pass_filter


def test_extra_1() -> None:
    """ 消融实验1 """
    print("---------------------- 消融实验1：不同姿态对模型的影响 ---------------------\n")
    total_dist_error = []
    total_dir_error = []
    dataset = PedestrianDataset(["test_extra_0"], window_size=200,
                                acceleration_filter=default_low_pass_filter, skip_len=5)
    for name, _ in dataset.loci.items():
        print("----------data: {} ----------".format(name))
        (dist_error, dir_error) = evaluate_model(dataset[name], pace_inference=pace_inference, compare=True)
        total_dist_error.append(dist_error)
        total_dir_error.append(dir_error)

    print("测试集总距离平均误差： {}".format(sum(total_dist_error) / len(total_dist_error)))
    print("测试集总方位角平均误差：{}".format(sum(total_dir_error) / len(total_dir_error)))
    print()
    print("测试集总距离方差： {}".format(np.asarray(total_dist_error).std()))
    print("测试集总方位角方差：{}".format(np.asarray(total_dir_error).std()))


def test_extra_2() -> None:
    """ 消融实验2 """
    print("---------------------- 消融实验2：不同设备对模型的影响 ---------------------\n")
    total_dist_error = []
    total_dir_error = []
    dataset = PedestrianDataset(["test_extra_1"], window_size=200,
                                acceleration_filter=default_low_pass_filter, skip_len=5)
    for name, _ in dataset.loci.items():
        print("----------data: {} ----------".format(name))
        (dist_error, dir_error) = evaluate_model(dataset[name], pace_inference=pace_inference, compare=True)
        total_dist_error.append(dist_error)
        total_dir_error.append(dir_error)

    print("测试集总距离平均误差： {}".format(sum(total_dist_error) / len(total_dist_error)))
    print("测试集总方位角平均误差：{}".format(sum(total_dir_error) / len(total_dir_error)))
    print()
    print("测试集总距离方差： {}".format(np.asarray(total_dist_error).std()))
    print("测试集总方位角方差：{}".format(np.asarray(total_dir_error).std()))


def test() -> None:
    """ 测试集 """
    print("---------------------- 测试集：模型总体性能评估 ---------------------\n")
    total_dist_error = []
    total_dir_error = []
    dataset = PedestrianDataset(["test_eval"], window_size=200,
                                acceleration_filter=default_low_pass_filter, skip_len=5)
    for name, _ in dataset.loci.items():
        print("----------data: {} ----------".format(name))
        (dist_error, dir_error) = evaluate_model(dataset[name], pace_inference=pace_inference, compare=True)
        total_dist_error.append(dist_error)
        total_dir_error.append(dir_error)

    print("测试集总距离平均误差： {}".format(sum(total_dist_error) / len(total_dist_error)))
    print("测试集总方位角平均误差：{}".format(sum(total_dir_error) / len(total_dir_error)))
    print()


if __name__ == '__main__':
    test()  # 测试集：模型总体性能评估
    test_extra_1()  # 消融实验1：不同姿态对模型的影响
    test_extra_2()  # 消融实验2：不同设备对模型的影响
