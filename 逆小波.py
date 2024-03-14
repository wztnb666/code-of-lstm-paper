import pandas as pd
import pywt

dataset = pd.read_excel('test2.xlsx',nrows=2000)

approximation = dataset.iloc[:, -2].values
details = [dataset.iloc[:, -1].values]

coeffs = [approximation] + details

# 通过逆小波将近似分量和细节分量合并回去
reconstructed_signal = pywt.waverec(coeffs, 'dmey')

cA = reconstructed_signal.astype('float64')
import xlwt
file = xlwt.Workbook('encoding = utf-8')
sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
sheet1.write(0, 0, "Predict_value")  # 第1行第3列
for i in range(len(cA)):
    sheet1.write(i + 1, 0, cA[i])
file.save('data/逆小波.xls')
