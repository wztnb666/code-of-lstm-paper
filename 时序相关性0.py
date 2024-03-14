import pandas as pd
from scipy.stats import spearmanr

# 假设你的数据存储在CSV文件中，使用pandas读取
data = pd.read_excel('test.xlsx')

# 你的目标列
target_column = 'grid1-loss'

# 其他列
other_columns = ['气温', '方位角', '云层不透明度', '露点温度',
         'DHI（太阳散射辐射指数）', 'DNI（太阳直接辐射指数）', 'GHI（太阳总水平辐射）', 'GTI（固定倾角辐射）',
         'GTI（跟踪倾角辐射）', '实际功率', '大气可降水量', '相对湿度',
         '地面气压', '高度10m风向', '高度10m风速', '天顶角',
         'grid1-load']

# 计算并打印每一列与目标列的相关性
for column in other_columns:
  correlation, _ = spearmanr(data[target_column], data[column])
  print(f'Correlation between {target_column} and {column}: {correlation:.3f}')
