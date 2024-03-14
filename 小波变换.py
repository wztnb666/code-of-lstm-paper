import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

# 假设我们有一列数据 data
data = pd.read_excel('test.xlsx')
Y= data.iloc[:,[-3]]

# 使用小波变换将数据分解为近似（低频）和细节（高频）分量
cA, cD = pywt.dwt(Y, 'dmey')

# cA 是近似分量，也就是低频分量
# cD 是细节分量，也就是高频分量

cA = cA.astype('float64')[:,0]
cD = cD.astype('float64')[:,0]
import xlwt
file = xlwt.Workbook('encoding = utf-8')
sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
sheet1.write(0, 0, "Predict_value")  # 第1行第3列
sheet1.write(0, 1, "True_value")  # 第1行第4列
for i in range(len(cA)):
    sheet1.write(i + 1, 0, cA[i])
    sheet1.write(i + 1, 1, cD[i])
file.save('data/小波.xls')

plt.figure()
plt.plot(cA,c='b',label='低频分量')
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('线损低频分量/MW', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.show()

plt.figure()
plt.plot(cD,c='r',label='高频分量')
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('线损高频分量/MW', fontsize=15)
plt.legend(fontsize=15, loc='upper right')
plt.show()
