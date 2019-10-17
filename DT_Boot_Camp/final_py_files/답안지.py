# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:15:12 2019

@author: user
"""

import numpy as np
import pandas as pd

happy = pd.read_csv("happy.csv")
happy1 = happy.copy()


np.sum(happy1.isna()) 
np.mean(happy1)

happy1.columns


happy1.Economy = np.where(happy1.Economy.isna(),
                  np.mean(happy.Economy),
                  happy.Economy)
happy1.Family = np.where(happy1.Family.isna(),
                  np.mean(happy.Family),
                  happy.Family)
happy1.Health = np.where(happy1.Health.isna(),
                  np.mean(happy.Health),
                  happy.Health)
happy1.Freedom = np.where(happy1.Freedom.isna(),
                  np.mean(happy.Freedom),
                  happy.Freedom)
happy1.Trust = np.where(happy1.Trust.isna(),
                  np.mean(happy.Trust),
                  happy.Trust)

happy1.dtypes

from sklearn.neighbors import LocalOutlierFactor 
happy2 = happy1.drop(columns = ["Rating", "Grade"])

from sklearn.neighbors import LocalOutlierFactor 
lof1 = LocalOutlierFactor()
lof1.fit(happy2)   

x =happy2.loc[lof1.negative_outlier_factor_>-2,:] 
y = happy[["Rating", "Grade"]][lof1.negative_outlier_factor_>-2] 

from sklearn.model_selection import train_test_split
tr_x, te_x, tr_y, te_y = train_test_split(x.drop(columns = "Score"), x.Score, test_size = 0.3, 
                                          random_state = 1234)


from sklearn.preprocessing import MinMaxScaler
minmax1 = MinMaxScaler()
tr_xs = tr_x.copy()
te_xs = te_x.copy()
minmax1.fit(tr_x)
tr_xs = minmax1.transform(tr_xs)
te_xs = minmax1.transform(te_xs)


from sklearn.linear_model import LinearRegression
lm_model=LinearRegression()
lm_model.fit(tr_xs, tr_y)
lm_model.coef_
lm_model.intercept_
lm_model.score(te_xs, te_y)


pred1=lm_model.predict(te_xs)
np.sum(pred1)
from sklearn.metrics import mean_squared_error
mean_squared_error(te_y, pred1)
lm_model.score(te_xs, te_y)


