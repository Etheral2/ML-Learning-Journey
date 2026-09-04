# 第 02 课 · 线性模型（Linear Model）
# 内容：线性模型 y = w*x，穷举 w 计算 MSE，画出代价曲线
# 依赖：numpy, matplotlib


import numpy as np
import matplotlib.pyplot as plt

# 准备训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型（线性模型 y = w*x）
def forward(x):
    return x * w


# 定义损失函数（单样本）
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 保存每个 w 及其对应的 MSE
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):      # 遍历 w = 0.0, 0.1, ... 4.0
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)            # 代价 = 损失的平均
    w_list.append(w)
    mse_list.append(l_sum / 3)

# 画「权重 w vs 代价 MSE」曲线
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
