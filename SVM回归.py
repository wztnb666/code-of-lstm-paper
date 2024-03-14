# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 创建训练数据和测试数据
# 加载数据（示例数据为CSV格式）
dataset = pd.read_excel('test2.xlsx',nrows=2000)
x = dataset.iloc[:,:-5]
y = dataset.iloc[:,[-5]]
# 初始化归一化器
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
# 对选定的列进行归一化
x = scalerX.fit_transform(x)
y = scalerY.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# 创建SVM回归模型并训练
svm_model = SVR(kernel='poly', C=1, epsilon=0.1)
svm_model.fit(x_train, y_train)

# 对测试数据进行预测
y_pred = svm_model.predict(x_test)
y_pred = y_pred[:,np.newaxis]

# 计算均方误差
# mse = mean_squared_error(y_test.values, y_pred)
# print("Mean squared error: ", mse)

y_pred = scalerY.inverse_transform(y_pred)
y_test = scalerY.inverse_transform(y_test)

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
file.save('data/总线损预测3-SVM回归.xls')

plt.figure()
plt.plot(y_test,c='b',label='True')
plt.plot(y_pred,c='r',label='Predict')
plt.legend()
plt.show()
