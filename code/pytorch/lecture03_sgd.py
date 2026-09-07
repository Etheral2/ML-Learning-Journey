
# 第 03 课 · 随机梯度下降（Stochastic Gradient Descent, SGD）
# 内容：SGD 用「单个样本的损失 loss」的导数逐样本更新权重
#       与 GD 的区别：GD 用全体样本 cost 的导数，SGD 用单样本 loss 的导数


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0                                     # 权重的初始猜测值


# 定义模型（线性模型 y = w*x）
def forward(x):
    return x * w


# 定义损失函数（单样本）
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 定义梯度函数（单样本 loss 的导数）
def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad                 # 每个样本更新一次权重
        print('\tgrad: ', x, y, grad)
        l = loss(x, y)

    print('progress:', epoch, 'w=', w, 'loss=', l)

print('Predict (after training)', 4, forward(4))
