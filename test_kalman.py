import numpy as np
from filterpy.kalman import KalmanFilter

dt = 1
sigma_a = 0.2
sigma_x = sigma_y = 3.

f = KalmanFilter(dim_x=6, dim_z=2)

# 转换矩阵F
f.F = np.array([[1., dt, 0.5 * dt * dt, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt * dt],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1.]])

# 过程噪声
f.Q = np.array([[0.25 * pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2), 0, 0, 0],
                [0.5 * pow(dt, 3), dt * dt, dt, 0, 0, 0],
                [0.5 * pow(dt, 2), dt, 1, 0, 0, 0],
                [0, 0, 0, 0.25 * pow(dt, 4), 0.5 * pow(dt, 3), 0.5 * pow(dt, 2)],
                [0, 0, 0, 0.5 * pow(dt, 3), dt * dt, dt],
                [0, 0, 0, 0.5 * pow(dt, 2), dt, 1]]) * sigma_a * sigma_a

# 测量噪声 , 必须是二维数组
f.R = np.array([[sigma_x * sigma_x, 0],
                [0, sigma_y * sigma_y]])

# 观测矩阵
f.H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]])
# 初始值
f.x = np.zeros(6)

# 协方差矩阵  默认是个单位矩阵 *500就行
f.P *= 500.

z = [[-393.66, 300.4], [-375.93, 301.78], [-351.04, 295.1]]

f.predict()
print("----init----")
print("x_pred:")
print(f.x)
print("p_pred:")
print(f.P)

for i in range(3):
    print("----iter:{}----".format(i))
    zi = np.array(z[i]).reshape(2, 1)
    # 更新参数
    f.update(zi)
    print("x_upd:")
    print(f.x)
    print("p_upd:")
    print(f.P)

    # 预测
    f.predict()
    print("K:")
    print(f.K)
    print("x_pred:")
    print(f.x)
    print("p_pred:")
    print(f.P)

