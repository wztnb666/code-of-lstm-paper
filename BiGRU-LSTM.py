import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

dataset = pd.read_excel('test.xlsx',nrows=2000)
X = dataset.iloc[:,:-2]
Y= dataset.iloc[:,[-2]]

scalerX = MinMaxScaler()
normalized_dataX = scalerX.fit_transform(X)
scalerY = MinMaxScaler()
normalized_dataY = scalerY.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(normalized_dataX, normalized_dataY, test_size=0.3, shuffle=False)

X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train).float()
y_test=torch.from_numpy(y_test).float()

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size=18, hidden_size=18, num_layers=1, batch_first=True)
        self.linear = nn.Linear(18, 1)

    def forward(self, input):
        # LSTM层的前向传播
        lstm_out, _ = self.lstm(input)
        output = self.linear(lstm_out)
        return output

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.bigru = nn.GRU(input_size=18, hidden_size=12, num_layers=2, batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(12, 1)

    def forward(self, x):
        bigru_out, _ = self.bigru(x)
        # 取最后一个时间步的输出
        out = self.fc(bigru_out)
        return out

class BiGRU(nn.Module):
    def __init__(self):
        super(BiGRU, self).__init__()

        self.bigru = nn.GRU(input_size=18, hidden_size=6, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(12, 1)

    def forward(self, x):
        bigru_out, _ = self.bigru(x)
        # 取最后一个时间步的输出
        out = self.fc(bigru_out)
        return out

class GRU_LSTM(nn.Module):
    def __init__(self):
        super(GRU_LSTM, self).__init__()

        self.gru = nn.GRU(input_size=18, hidden_size=18, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=18, hidden_size=18, num_layers=1, batch_first=True)
        self.fc = nn.Linear(18, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        lstm_out, _ = self.lstm(gru_out)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out)
        return out

class BiGRU_LSTM(nn.Module):
    def __init__(self):
        super(BiGRU_LSTM, self).__init__()

        self.bigru = nn.GRU(input_size=18, hidden_size=32, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.lstm = nn.LSTM(input_size=32 * 2, hidden_size=64, num_layers=1,
                            batch_first=True)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        bigru_out, _ = self.bigru(x)
        lstm_out, _ = self.lstm(bigru_out)
        # 取最后一个时间步的输出
        out = self.fc1(lstm_out)
        return out

# 创建模型实例
model = GRU()
#model = torch.nn.DataParallel(model)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.05)

#0.005 BiGRU_LSTM


# 训练模型
for epoch in range(100):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch,loss.item())

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred = scalerY.inverse_transform(y_pred)
y_test = scalerY.inverse_transform(y_test)

# plt.figure()
# plt.plot(y_pred,c='r', label='Predict')
# plt.plot(y_test,c='b', label='True')
# plt.legend()
# plt.show()

mae = np.mean(np.abs(y_pred - y_test))
mape = np.mean(np.abs((y_pred - y_test) / y_test))
rmse = np.sqrt(np.mean(np.power(y_pred - y_test, 2)))
column_vector = np.full((len(y_test), 1), np.mean(y_test))
R2 = 1-(np.sum(np.power(y_pred - y_test, 2))/np.sum(np.power(column_vector - y_test, 2)))

y_pred = y_pred.astype('float64')
y_test = y_test.astype('float64')
import xlwt
file = xlwt.Workbook('encoding = utf-8')
sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
sheet1.write(0, 0, "Predict_value")  # 第1行第3列
sheet1.write(0, 1, "True_value")  # 第1行第4列
sheet1.write(0, 3, "mae")  # 第1行第5列
sheet1.write(0, 4, "mape(%)")  # 第1行第5列
sheet1.write(0, 5, "rmse")  # 第1行第5列
sheet1.write(0, 6, "R2")  # 第1行第5列
for i in range(len(y_test)):
    sheet1.write(i + 1, 0, y_pred[i][0])
    sheet1.write(i + 1, 1, y_test[i][0])
sheet1.write(1, 3, mae.item())
sheet1.write(1, 4, mape.item()*100)
sheet1.write(1, 5, rmse.item())
sheet1.write(1, 6, R2.item())
file.save('data/总线损预测3-GRU_LSTM.xls')

# 输出预测结果
plt.figure()
plt.plot(y_test, c='b', label='实际值')
plt.plot(y_pred, c='r', label='预测值')
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('线路损耗/MW', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.show()