import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

transformer_capacity = [500, 750, 1000, 1500, 2000]#公变变压器总容量
line_length = [5, 7, 10, 15, 20]
line_type = ['type1', 'type2', 'type1', 'type2', 'type1']
capacitor_capacity = [100, 150, 200, 250, 300]#无功补偿装置容量
line_loss = [50, 70, 90, 110, 130]

# 创建DataFrame
df = pd.DataFrame({
  'transformer_capacity': transformer_capacity,
  'line_length': line_length,
  'line_type': line_type,
  'capacitor_capacity': capacitor_capacity,
  'line_loss': line_loss
})

# 对于分类变量，我们可以先将其转换为数值变量，例如通过独热编码
df = pd.get_dummies(df)

# 计算相关系数
correlation_matrix = df.corr()
print(correlation_matrix)
# 绘制相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
