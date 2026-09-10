# 第 07 课 · 多维输入（Multiple Dimension Input）
# 内容：糖尿病数据集（8 个特征 → 二分类）
#       多层神经网络：Linear(8,6) → Linear(6,4) → Linear(4,1)
#       每层后接 Sigmoid（非线性激活）
# 依赖：diabetes.csv.gz（与本目录同级的父目录）

import numpy as np
import torch

# 1. 准备数据集
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])      # 前 8 列是特征
y_data = torch.from_numpy(xy[:, [-1]])     # 最后一列是矩阵


#  2. 用类设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid() #sigmoid加入非线性模块

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

#  3. 构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#  4. 训练循环
for epoch in range(100):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()


# ============================================================
# 练习：换用 ReLU 激活函数（课程 Exercise 片段）
# ============================================================
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear1 = torch.nn.Linear(8, 6)
#         self.linear2 = torch.nn.Linear(6, 4)
#         self.linear3 = torch.nn.Linear(4, 1)
#         self.activate = torch.nn.ReLU()          # 换 ReLU
#
#     def forward(self, x):
#         x = self.activate(self.linear1(x))
#         x = self.activate(self.linear2(x))
#         x = self.activate(self.linear3(x))
#         return x
#
# model = Model()
