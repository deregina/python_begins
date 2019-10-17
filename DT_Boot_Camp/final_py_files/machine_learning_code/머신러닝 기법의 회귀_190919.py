# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:25:05 2019

@author: user
"""

# < 머신러닝 기법의 회귀 >

# (1) 데이터 분리 # train / test
# (2) 데이터 단위 맞추기
# (3) 식 도출 (학습)
# (4) 검정 (test) --> 추정값 나옴
# (5) 변수선택기법 / 최적변수 찾기 / 변수축약기법
# (6) 검정 
#     (underpitting --> 너무 적은 변수 사용, 예측력 낮음 
#      overpitting --> 학습용 데이터와 똑같으면 예측력이 좋은데
#                      이외의 데이터에서는 예측력이 나쁨, 
#                      학습용 데이터에 맞추어서 너무 많은 변수를 사용)

df.shape
# import scikit learn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
xx = df.drop(columns = "MEDV")
yy = df.MEDV

tr_x, te_x, tr_y, te_y = train_test_split(xx, yy, test_size = 0.3, random_state = 1234)

xx.shape             # (506, 13)
tr_x.shape           # (354, 13) --> xx data의 70%
te_x.shape           # (152, 13) --> xx data의 30%

type(xx)
type(tr_x)

tr_x.describe()

from sklearn.preprocessing import minmax_scale, MinMaxScaler
# minimum ~ maximum scale 맞춰주기
# 범주형 데이터는 처리 방법을 따로 정하고 수치형만 적용해줌
minmax = MinMaxScaler()
tr_xs = minmax.fit_transform(tr_x)
te_xs = minmax.transform(te_x)
# fit은 scale을 minmax 안에 저장하는 역할을 함
# 두 번째 실행할 때는 fit을 안 넣어서 아까 scale 정보를 이용함

type(tr_xs)
np.min(tr_xs, axis = 0)
np.max(tr_xs, axis = 0)

from sklearn.linear_model import LinearRegression
# Linear Regression : 선형회귀분석
lm_mod = LinearRegression()
# LinearRegression class의 모든 기능을 lm_mod라는 변수 안에 넣음
lm_mod.fit(tr_xs, tr_y)
lm_mod.coef_
# y = bx + a에서 b에 해당하는 값
lm_mod.intercept_
# y = bx + a에서 a에 해당하는 값
pred = lm_mod.predict(te_xs)
pred
# lm_mod에 회귀식이 들어있으므로 검정 data를 넣어서 y의 예측값들을 받음
te_y - pred
# y 예측값과 실제 y 값의 차이를 구함
np.mean((te_y - pred)**2)
# MSE(Error, 오차의 분산)를 구함
lm_mod.score(te_xs, te_y)
# R square 값을 구함, R^2가 클수록 회귀계수가 큼
np.argmax(lm_mod.coef_)
te_x.columns.values[5]


























