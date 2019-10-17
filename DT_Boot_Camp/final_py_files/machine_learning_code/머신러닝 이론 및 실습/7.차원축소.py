# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:13:21 2019

@author: pns
"""
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
                    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()
X, _ = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

sc = StandardScaler()                      ### 데이터 표준화 
X_std = sc.fit_transform(X)

winePCA = PCA(n_components = 2)            ### PCA 수행
res01 = winePCA.fit_transform(X_std)

wineKM = KMeans(n_clusters = 3)            ### PCA 결과를 토대로 K-means 수행  
res02 = wineKM.fit(res01)

cluster01 = res01[res02.labels_ == 0, :]
cluster02 = res01[res02.labels_ == 1, :]
cluster03 = res01[res02.labels_ == 2, :]

fig = plt.figure(figsize = (8, 8))
plt.scatter(x = cluster01[:, 0], y = cluster01[:, 1], c="r", alpha=0.7, label = "cluster0")
plt.scatter(x = cluster02[:, 0], y = cluster02[:, 1], c="g", alpha=0.7, label = "cluster1")
plt.scatter(x = cluster03[:, 0], y = cluster03[:, 1], c="b", alpha=0.7, label = "cluster2")
plt.legend(loc='upper left')
plt.show()


### PCA 성분을 3개로 설정 ###

winePCA = PCA(n_components = 3)            ### PCA 수행
res03 = winePCA.fit_transform(X_std)

wineKM = KMeans(n_clusters = 3)            ### PCA 결과를 토대로 K-means 수행  
res04 = wineKM.fit(res03)

cluster04 = res03[res04.labels_ == 0, :]
cluster05 = res03[res04.labels_ == 1, :]
cluster06 = res03[res04.labels_ == 2, :]

fig = plt.figure(figsize = (8, 8))
ax = Axes3D(fig)
ax.scatter(xs= cluster04[:, 0], ys= cluster04[:, 1], zs = cluster04[:, 2], c = "r", label = "cluster0")
ax.scatter(xs= cluster05[:, 0], ys= cluster05[:, 1], zs = cluster05[:, 2], c = "g", label = "cluster1")
ax.scatter(xs= cluster06[:, 0], ys= cluster06[:, 1], zs = cluster06[:, 2], c = "b", label = "cluster2")
plt.legend(loc='upper left')
plt.show()




