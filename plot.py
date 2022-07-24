import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

# scr-x scr-y   tcp-height   arm-x  ...      arm-z   prd-arm-x   prd-arm-y   prd-arm-z
df = pd.read_csv('data/calibrate-xy.csv')

X = df['scr-x']
Y = df['scr-y']

# Difference between actual values and predicted values.
dX = df['arm-x'] - df['prd-arm-x']
dY = df['arm-y'] - df['prd-arm-y']
dZ = df['arm-z'] - df['prd-arm-z']

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.plot(X,Y,dX,marker="o",linestyle='None')
ax.plot(X,Y,dY,marker="X",linestyle='None')
ax.plot(X,Y,dZ,marker="+",linestyle='None')

plt.show()