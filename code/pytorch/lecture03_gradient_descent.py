
# 第 03 课 · 梯度下降（Gradient Descent）
# 内容：全量梯度下降，用「所有样本的代价 cost」的导数更新权重
#       更新公式：w = w - α * ∂cost/∂w

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0                                     # 权重的初始猜测值


# 定义模型（线性模型 y = w*x）
def forward(x):
    return x * w


# 定义代价函数（MSE，全体样本的平均）
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# 定义梯度函数（MSE 对 w 的导数）
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val                   # 更新权重（学习率 0.01）
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Predict (after training)', 4, forward(4))
