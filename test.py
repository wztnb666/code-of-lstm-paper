#=============================================#
#https://blog.csdn.net/m0_65157892/article/details/129502566
#https://blog.csdn.net/weixin_42163563/article/details/119715312
#=============================================#
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_excel('test.xlsx')
x = data.iloc[:,:-3]
y = data.iloc[:,[-1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

#  建模
forest = RandomForestRegressor(
    n_estimators=50,
    random_state=1,
    n_jobs=None)
forest.fit(x_train, y_train)

score = forest.score(x_test, y_test)
print('随机森林模型得分： ', score)
y_validation_pred = forest.predict(x_test)

# 验证集结果输出与比对
plt.figure()
plt.plot(np.arange(874), y_test[:874], "b", label="True value")
plt.plot(np.arange(874), y_validation_pred[:874], "r", label="Predict value")
plt.title("True value And Predict value")
plt.legend()
plt.show()