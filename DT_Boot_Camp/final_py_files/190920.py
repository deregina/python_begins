
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:53:12 2019

@author: user
"""

import pandas as pd
import numpy as np


# (1) 데이터 로딩 및 탐색

boston = pd.read_csv("boston.csv")
boston.columns
boston.dtypes
boston1 = boston.copy()
boston1["A"] = "A"
boston1.head()
boston1.dtypes
boston1.dtypes == "float64"
boston1.dtypes != "object"
boston1.loc[:, boston1.dtypes != "object"]
# NA값을 채울 때,
# boston1.loc[:, boston1.dtypes != "object"].apply()
boston1.describe()





# (2) 회귀분석[다중회귀, 입력/종속변수]
# 회귀분석을 통해서 NA값(결측치)을 조금 더 정확하게 예측하고 채울 수 있음

np.sum(boston.isna())

xx = boston1.drop(columns = ["medv", "A"])
yy = boston1["medv"]





# (3) 이상치 존재 유무 확인
# NA가 없는 데이터 or NA 보정 후 수치형 데이터 중에서만 이상치 제외
# a. 회귀식을 구해 모듈을 통해 이상치 체크 --> statsmodels
# b. 회귀식 없이 데이터만 보고 이상치 찾기 --> sklearn.neighbors

from sklearn.neighbors import LocalOutlierFactor
lof1 = LocalOutlierFactor(n_neighbors = 5)
# 주변 5개의 점들과 거리를 계산해서 
lof1.fit(xx)
# 아웃라이어를 골라냄
lof1.negative_outlier_factor_
# 음수로 나옴. 숫자가 작을수록 더 아웃라이어

lof1.negative_outlier_factor_ > -2
# -2 ~ -2.5가 일반적
xx1 = xx.loc[lof1.negative_outlier_factor_ > -2, :]
xx1.shape
# 506개 행에서 21행이 사라짐
yy1 = yy[lof1.negative_outlier_factor_ > -2]





# (4) 학습용 / 테스트용 데이터셋 분리

from sklearn.model_selection import train_test_split
tr_x, te_x, tr_y, te_y = train_test_split(xx1, yy1, train_size = 0.3,
                                          random_state = 100)
type(tr_x)
tr_x.shape





# (5) 단위 표준화 (Scale 맞추기)

from sklearn.preprocessing import MinMaxScaler
minmax1 = MinMaxScaler()

x_list = tr_x.columns.drop("chas")

minmax1.fit(xx1[x_list])
# MinMixScaler가 "chas"열(0, 1의 범주데이터)을 제외하고 데이터 학습
tr_xs = tr_x.copy()
tr_xs.loc[:, x_list] = minmax1.transform(tr_xs.loc[:, x_list])
# tr_xs.loc[:, x_list]의 값을 0~1로 변환
np.min(tr_xs, axis = 0)
np.min(tr_xs, axis = 1)

te_xs = te_x.copy()
te_xs.loc[:, x_list] = minmax1.transform(te_xs.loc[:, x_list])





# (6) 회귀분석 적용

from sklearn.linear_model import LinearRegression
lm_model = LinearRegression()
lm_model.fit(tr_xs, tr_y)
lm_model.intercept_
lm_model.coef_





# (7) 검정 및 성능 평가

pred = lm_model.predict(te_xs)
r_square = lm_model.score(te_xs, te_y)

from sklearn.metrics import mean_squared_error
mean_squared_error(te_y, pred)





# (8) 변수 선택

from sklearn.feature_selection import RFE
rfe1 = RFE(estimator = lm_model, n_features_to_select = 4)
rfe1.fit(tr_xs, tr_y)
rfe1.get_support()
type(tr_xs)
tr_xs.columns.values[rfe1.get_support()]

tr_xs4 = rfe1.transform(tr_xs)
tr_xs4[0:3, :]
tr_xs4.shape

lm_model2 = LinearRegression()
lm_model2.fit(tr_xs4, tr_y)

te_xs4 = rfe1.transform(te_xs)
pred2 = lm_model2.predict(te_xs4)
lm_model2.score(te_xs4, te_y)
mean_squared_error(te_y, pred2)













































