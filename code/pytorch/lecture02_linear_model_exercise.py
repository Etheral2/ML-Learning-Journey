# 第 02 课 · 线性模型（Linear Model）- 作业练习
# 内容：线性模型 y = w*x+b，穷举 w 计算 MSE，画出代价曲线
# 依赖：numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #两个参数画3D图

# 准备训练集
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


# 定义模型（线性模型 y = w*x +b）
def forward(x,w,b):
    return x * w + b


# 定义损失函数（单样本）
def loss(x, y, w, b):
    y_pred = forward(x,w,b)
    return (y_pred - y) * (y_pred - y)


# 保存每个 w 及其对应的 MSE
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1): # 遍历 w = 0.0, 0.1, ... 4.0
    #print('w=', w)
    for b in np.arange(0.0, 4.1, 0.1): # 遍历 b = 0.0, 0.1, ... 4.0
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val,w,b)
            loss_val = loss(x_val, y_val,w,b)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        mse = l_sum / 3
        #print('MSE=', mse)  # 代价 = 损失的平均
        b_list.append(b)
        w_list.append(w)
        mse_list.append(l_sum / 3)

# 画「权重 w vs 代价 MSE」曲线
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# 转成网格用于曲面绘图
w_arr = np.array(w_list)
b_arr = np.array(b_list)
mse_arr = np.array(mse_list)

# 因为是按顺序遍历，reshape成网格形状
grid_size = len(np.arange(0.0,4.1,0.1))
W = w_arr.reshape(grid_size, grid_size)
B = b_arr.reshape(grid_size, grid_size)
MSE = mse_arr.reshape(grid_size, grid_size)

surf = ax.plot_surface(W, B, MSE, cmap="viridis", edgecolor='none')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE Loss')
ax.set_title("3D Loss Surface y=w*x+b")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
