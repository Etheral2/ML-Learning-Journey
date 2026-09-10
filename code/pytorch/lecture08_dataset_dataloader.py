# ============================================================
# 第 08 课 · Dataset 和 DataLoader
# 内容：自定义 Dataset + DataLoader，实现 Mini-Batch 训练
#       术语：Epoch（全部样本一次前+反）、Batch-Size（一次前反的样本数）、
#             Iteration（用 batch_size 个样本做一次 pass 的次数）
# 依赖：diabetes.csv.gz（与本目录同级的父目录）
# 注意：Windows 下 num_workers > 0 需用 if __name__ == '__main__' 包裹
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- 自定义 Dataset（继承抽象类 Dataset） ----------
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):          # 支持 dataset[index] 索引
        return self.x_data[index], self.y_data[index]

    def __len__(self):                     # 返回数据集长度
        return self.len


# ---------- 构造 Dataset 和 DataLoader ----------
dataset = DiabetesDataset('../diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,   # 每批 32 个样本
                          shuffle=True,    # 训练集打乱
                          num_workers=2)   # 多进程加载


# ---------- 模型 ----------
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1. 准备数据
            inputs, labels = data
            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            # 4. Update
            optimizer.step()
