import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有一个时间序列数据集 dataset
dataset = pd.read_excel('test.xlsx')
X = dataset.iloc[:,:-5]
Y= dataset.iloc[:,[-5]]
X = torch.from_numpy(X.values).float()
Y = torch.from_numpy(Y.values).float()

class NARMA(nn.Module):
  def __init__(self):
    super(NARMA, self).__init__()
    self.rnn = nn.RNN(17, 17, batch_first=True)
    self.fc = nn.Linear(17, 1)

  def forward(self, x):
    #h0 = torch.zeros(1, len(x), self.hidden_size)  # initial hidden state
    x = x.reshape((x.shape[0],1, x.shape[1]))
    out, _  = self.rnn(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
    out = self.fc(out[:, -1, :])  # decode the hidden state of the last time step
    return out

# 建立模型
model = NARMA()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10000):
    outputs = model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 打印权重
for name, param in model.named_parameters():
  if param.requires_grad:
    print(name, param.data)
