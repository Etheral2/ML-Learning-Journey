# 第 06 课 · 逻辑回归（Logistic Regression）
# 内容：二分类。模型输出「属于某类的概率」
#       模型：y = σ(wx + b)，σ 为 Sigmoid
#       损失：BCE（二元交叉熵）
# 依赖：numpy, matplotlib（画图）

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. 准备数据集 ----------
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])     # 二分类标签 0/1


# ---------- 2. 用类设计模型 ----------
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  # 与05课的不同：线性层后接 Sigmoid，将输出值控制在0和1之间
        return y_pred


model = LogisticRegressionModel()

# ---------- 3. 构造损失函数和优化器 ----------
criterion = torch.nn.BCELoss(size_average=False)  # 二元交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ---------- 4. 训练循环 ----------
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------- 画图观察分类概率曲线 ----------
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')       # 0.5 分界线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
