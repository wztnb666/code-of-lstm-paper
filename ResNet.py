import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

# 构建ResNet模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_blocks)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_blocks)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(10, 10, kernel_size=1)
        self.conv2 = nn.Conv1d(10, 1, kernel_size=1)
        self.bn = nn.BatchNorm1d(18)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 10, 18)
        self.linear = nn.Linear(18, 10)

    def make_layer(self, block, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(out_channels, out_channels, num_blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.conv2(out)
        out = self.linear(out).transpose(0,1)
        return out

# 读取数据
dataset = pd.read_excel('test2.xlsx',nrows=2000)
X = dataset.iloc[:,:-5]
y = dataset.iloc[:,[-5]]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的ResNet模型
model = ResNet()

# 选择优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.MSELoss()

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
  for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    pred = []
    for i, (inputs, labels) in enumerate(test_loader):
        y_pred = model(inputs)
        pred.append(y_pred)
result = torch.cat(pred, dim=0)

# 输出预测结果
plt.figure()
plt.plot(y_test, c='b', label='实际值')
plt.plot(result, c='r', label='预测值')
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('线路损耗/MW', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.show()
