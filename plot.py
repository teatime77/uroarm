import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

# scr-x scr-y   tcp-height   arm-x  ...      arm-z   prd-arm-x   prd-arm-y   prd-arm-z
df = pd.read_csv('data/calibrate-xy.csv')
# print(df['scr-x'])
print(df['scr-y'])

# 3D散布図でプロットするデータを生成する為にnumpyを使用
# X = np.array([i for i in range(1,100)]) # 自然数の配列
# Y = np.sin(X) # 特に意味のない正弦
# Z = np.sin(Y) # 特に意味のない正弦
X = df['scr-x']
Y = df['scr-y']
dX = df['arm-x'] - df['prd-arm-x']
dY = df['arm-y'] - df['prd-arm-y']
dZ = df['arm-z'] - df['prd-arm-z']

# グラフの枠を作成
fig = plt.figure()
ax = Axes3D(fig)

# X,Y,Z軸にラベルを設定
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# .plotで描画
ax.plot(X,Y,dX,marker="o",linestyle='None')
ax.plot(X,Y,dY,marker="X",linestyle='None')
ax.plot(X,Y,dZ,marker="+",linestyle='None')

# 最後に.show()を書いてグラフ表示
plt.show()