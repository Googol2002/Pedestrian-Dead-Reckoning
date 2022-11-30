from evaluate.test_evalutator import evaluate_model
from pace_predictor.predict_pace import magic_pace_inference
from pace_predictor.acc_pace_inference import pace_inference
from pedestrian_data import PedestrianDataset, default_low_pass_filter

""" 我们收集数据的测试集评估 """


def test() -> None:
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


if __name__ == '__main__':
    test()
