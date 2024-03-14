import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

data1 = pd.read_excel('test2.xlsx',dtype='float32').values
# data2 = pd.read_excel('data/总线损预测3.xls',dtype='float32').values

# plt.figure()
# plt.plot(data1[:,[0]], c='k', label='实际值')
# plt.plot(data1[:,[1]], c='r', label='GRU')
# plt.plot(data1[:,[2]], c='b', label='LSTM')
# plt.plot(data1[:,[3]], c='g', label='BiGRU')
# plt.plot(data1[:,[4]], c='m', label='GRU-LSTM')
# plt.plot(data1[:,[4]], c='y', label='BiGRU-LSTM')
# plt.plot(data2[:,1], c='b', label='低频实际值')#c='m'
# plt.plot(data2[:,0], c='r', label='低频预测值')#c='g'
# plt.plot(data2[:,1], c='k', label='实际值')
# plt.plot(data2[:,0], c='r', label='总预测值')

# 创建一个figure对象
fig, axs = plt.subplots(5)

# 在每个subplot上绘图
axs[0].plot(data1[:,[-5]], c='r', label='真实值')
axs[0].legend(loc='upper right')
axs[1].plot(data1[:,[-4]], c='b', label='低频分量（第三层）')
axs[1].legend(loc='upper right')
axs[2].plot(data1[:,[-3]], c='g', label='高频分量（第三层）')
axs[2].legend(loc='upper right')
axs[3].plot(data1[:,[-2]], c='g', label='高频分量（第二层）')
axs[3].legend(loc='upper right')
axs[4].plot(data1[:,[-1]], c='g', label='高频分量（第一层）')
axs[4].legend(loc='upper right')
# 显示图像
plt.show()




# 设置坐标轴刻度字号
# plt.tick_params(axis='both', labelsize=15)
# # 设置坐标轴标签和标题
# # plt.xlabel('x', fontsize=12)
# plt.ylabel('线路损耗/MW', fontsize=15)
# plt.legend(fontsize=15, loc='upper right', ncol=6)
# plt.show()