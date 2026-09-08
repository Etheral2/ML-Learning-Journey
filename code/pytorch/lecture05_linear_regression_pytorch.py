# 第 05 课 · 用 PyTorch 实现线性回归（Linear Regression with PyTorch）
# 内容：PyTorch 编程「四步法」（全课程通用范式）
#       1. 准备数据集
#       2. 用类设计模型（继承 nn.Module）
#       3. 构造损失函数和优化器（PyTorch API）
#       4. 训练循环（forward → backward → update）

import torch

# 1. 准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 2. 用类设计模型（继承 nn.Module）
class LinearModel(torch.nn.Module):
    def __init__(self): #初始化函数
        super(LinearModel, self).__init__()  #【必须写】！
        self.linear = torch.nn.Linear(1, 1) #输入维度1，输出维度1
        # 线性层，自带 weight 和 bias

    def forward(self, x):
        y_pred = self.linear(x) # 执行矩阵乘法：y_pred = x @ W.T + b
        return y_pred


model = LinearModel() #实例化

# 3. 构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)  #不求平均值，影响不大
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # model.parameters() 会取出模型里所有的矩阵参数（W 和 b）
                                                          # lr=0.01 是步长(learning rate)

#  4. 训练循环
for epoch in range(1000):
    y_pred = model(x_data)                 # Forward：前向预测
    loss = criterion(y_pred, y_data)       # 计算损失
    print(epoch, loss.item()) #.item()改为标量

    optimizer.zero_grad()                  # 清空梯度！！！！（否则累加）
    loss.backward()                        # Backward：反向传播
    optimizer.step()                       # Update：更新参数

#  输出权重与偏置
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#  测试模型
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
