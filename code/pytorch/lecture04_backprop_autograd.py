
# 第 04 课 · 反向传播（Back Propagation）
# 内容：用 PyTorch 的 Tensor + autograd 自动求梯度实现线性模型
#       关键点：
#         1. w.requires_grad = True  开启自动求导
#         2. l.backward()            反向传播求梯度
#         3. w.grad                  访问梯度
#         4. 梯度会累加，更新后必须 w.grad.data.zero_() 清零

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])                     # 权重是一个 Tensor
w.requires_grad = True                      # 需要计算梯度，必须设为 True，默认不计算梯度


# 定义模型（线性模型 y = w*x）
def forward(x):
    return x * w


# 定义损失函数（单样本）
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)                      # 前向，计算损失
        l.backward()                        # 反向，自动求梯度
        print('\tgrad:', x, y, w.grad.item()) # item取标量
        w.data = w.data - 0.01 * w.grad.data  # 用梯度更新权重（.data 绕开计算图）

        w.grad.data.zero_()                 # 梯度会累加，更新后必须清零！！

    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
