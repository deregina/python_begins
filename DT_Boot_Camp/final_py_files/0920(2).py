# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:52:53 2019

@author: user
"""

import pandas as pd
import numpy as np

# [1] 데이터 탐색 
boston = pd.read_csv("boston.csv")
boston.columns
boston.dtypes
boston1 = boston.copy()
boston1["A"] = "A" # A라는 열을 추가해주고 그 내용은 A 
boston1.head()
boston1.dtypes
boston1.dtypes == "object"
boston1.loc[:, boston1.dtypes!= "object"]
# object 타입이 아닌 애들로 (행은 전부 포함)

# [2] 다중회귀분석 - 입력, 종속 변수의 확인  
boston1.describe() # scale이 동일한지 확인한다 / 데이터의 단위가 다소 차이가 있음
# n.a 데이터 확인 및 처리 
boston.isna() # n.a 여부 바로 확인 가능 / notna() n.a 아닌 개수 
np.sum(boston.isna()) 

xx = boston1.drop(columns = ["medv", "A"])
y = boston1["medv"]

# [3] 이상치 존재 유무 확인 
# a. 회귀식을 구한 다음, 그 모듈 안에 이상치를 채크 (statsmodels)
# b. 회귀식 없이 순수 데이터만 가지고 이상치 찾기 

# 최근접이웃 KNN - 나를 기준으로 가까운 애들을 찾음. "거리공식" - 유클리드 거리 
# 가까운 애들의 Y label을 내가 쓸게요~ 

from sklearn.neighbors import LocalOutlierFactor 
lof1 = LocalOutlierFactor(n_neighbors=5)
# 이웃의 개수, 
lof1.fit(xx) # fit(), 찾아주세용 / 거리를 구할떄 수치형 데이터를 넣어야 한다. 범주형 입력 불가  
len(lof1.negative_outlier_factor_)  # 마이너스 쪽으로 더 가면 갈수록 outlier 
# negative_outlier_factor_ 이상치를 찾아준다 
lof1.negative_outlier_factor_>-2 
# -2 정도면 2δ를 넘어가는 정도임, deadline - 일반화 시킨 거리이기 때문에 
# 만약 더 크게 하고 싶으면 -2 정도임  
boston.loc[lof1.negative_outlier_factor_>-2,:] # 모든 열에 대해서 
xx1 = xx.loc[lof1.negative_outlier_factor_>-2,:] 
xx1.shape
y1 = y[lof1.negative_outlier_factor_>-2] # 이상치에 해당하는 열들이 삭제되는 거니까 
# xx1으로 결정변수 이상치를 제거했으므로 y 종속변수에서 이제 삭제해야함 
len(y1)

# (1) na가 없는 수치형 데이터만 이용해서 이상치 유무 체크 
# (2) na가 존재하면 na를 보정한 후 이상치 유무 체크 
# (3) na를 어떻게 보정할 것인지 고려 

from sklearn.model_selection import train_test_split

## test용, 학습용으로 데이터셋 분리 
tr_x, te_x, tr_y, te_y = train_test_split(xx1,y1, test_size=0.3, random_state=100)

type(tr_x)
tr_x.shape

## 단위 표준화 scale 통일 
from sklearn.preprocessing import MinMaxScaler
# MinMaxScaler은 수치형에만 적용할 수 있다. 
minmax1=MinMaxScaler()

x_list=tr_x.columns.drop("chas")
x_list.shape

minmax1.fit(xx1[x_list])
tr_xs=tr_x.copy()
tr_xs.loc[:, x_list]=minmax1.transform(tr_xs.loc[:, x_list])

np.mean(tr_xs, axis=0)

te_xs=te_x.copy()
te_xs.loc[:, x_list]=minmax1.transform(te_xs.loc[:, x_list])

np.mean(te_xs, axis=0)


## 범주형 데이터 dummy 로 처리 0,1 
# chas : 강의 유무 변수, 1: 강의 있음 / 0: 강의 없음 (dummy 처리)

## (6) 회귀분석 적용 
from sklearn.linear_model import LinearRegression

lm_model=LinearRegression()
lm_model.fit(tr_xs, tr_y)
lm_model.coef_
lm_model.intercept_

## (7) 예측 및 검정 
pred1=lm_model.predict(te_xs) # 만들어진 coef와 intercept로 식을 만들어서 입력값을 도출 
r_square=lm_model.score(te_xs, te_y)


te_xs.shape
te_y.shape

pred1.shape

from sklearn.metrics import mean_squared_error
mean_squared_error(te_y, pred1)





# (8) 변수 선택

from sklearn.feature_selection import RFE
rfe1 = RFE(estimator = lm_model, n_features_to_select = 5)
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

# 변수 4개보다 변수 5개가 mean square 값이 더 높음 --> 더 효율적





# (9) 변수 축약

from sklearn.decomposition import PCA

pca1 = PCA(n_components = 7)
pca1.fit(tr_xs)
pca1.components_
# Beat값(회귀계수)의 모음집
pca1.explained_variance_
# eigenvalue
# [0.43637962, 0.10747217, 0.07678906, 0.04844739, 0.03404881]
# 전체 데이터 중 첫번째 변수 43%, 두번째 10%... (전체 67%) 설명

# pca1 = PCA(n_components = 5)
# pca1.fit(tr_xs)
# pca1.components_
# pca1.explained_variance_
# 변수 5 -> 7 lm_model3.score가 57.0% --> 72.9%로 상승
# 많은 변수 중 2개 정도만 넣었는데 score가 90% 나오면 다른 변수 필요 없음


tr_xs5 = pca1.transform(tr_xs)
te_xs5 = pca1.transform(te_xs)
tr_xs.shape
tr_xs5.shape

lm_model3 = LinearRegression()
lm_model3.fit(tr_xs5, tr_y)
pred3 = lm_model3.predict(te_xs5)
lm_model3.score(te_xs5, te_y)

# pd.DataFrame(pca1.components_).to_excel("decom1.xlsx")

mean_squared_error(te_y, pred3)





#################################################################
# 범주형 데이터 ("chas" 열)
# 0, 1 / 1, 2

# y = a + 0.6 * 1 + 0.3 * 2 + 0.1 * "chas"변수
# chas : 0, 1, 2, 3
# One-Hot : a + 0.6 * 1 + 0.3 * 2 + 
#           0.4 * D1 + 0.5 * D2 + 0.3 * D3 + 0.1 * D4
# One-Hot : 0.4D1, 0.5D2, 0.3D3, 0.1D4
#             1,     0,     0,     0
#             0,     1,     0,     0
#             0,     0,     1,     0     
#             0,     0,     0,     1      
# chas는 값누적이 아니라 해당 가중치*값에 대한 값만 기존 식에 들어감

# dummy : a + 0.6 * 1 + 0.3 * 2 + 0.4 * D1
# Group0만 포함된 식
# D1이 표에서 사라짐
# Group1, Group2...의 경우 그냥 변동량을 더해주게 됨
# Dummy : 0.5D2, 0.3D3, 0.1D4
#           0,     0,     0
#           1,     0,     0
#           0,     1,     0     
#           0,     0,     1      


survey = pd.read_csv("survey.csv")
survey.dtypes
survey.Sex = survey.Sex.astype("category")
survey.dtypes

# one-hot
survey2 = pd.get_dummies(survey)
# dummies
survey3 = pd.get_dummies(survey, drop_first = True)

survey.columns.values
x_list2 = ['Wr.Hnd', 'NW.Hnd', 'Exer', 'Pulse']

survey4 = pd.get_dummies(survey[x_list2], drop_first = True)
survey5 = survey4.dropna()

lm_model_dum = LinearRegression()
lm_model_dum.fit(survey5.drop(columns = "Pulse"), survey5["Pulse"])
lm_model_dum.coef_
lm_model_dum.intercept_


#################################################################
# 분류기법
sonar = pd.read_csv("Sonar.csv")
sonar.head()
sonar.shape
sonar.columns

xx = sonar.drop(columns = "Class")
yy = sonar["Class"]

tr_x, te_x, tr_y, te_y = train_test_split(xx, yy, test_size = 0.3, random_state = 200)

from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors = 3)
# 데이터가 뒤섞여 있으면 n_neighbors 수를 낮추는 것이 좋음

knn1.fit(tr_x, tr_y)
pred5 = knn1.predict(te_x)
pred5 = knn1.predict_proba(te_x)
# [0.66666667, 0.33333333],[1.        , 0.        ]
# 이웃 점 3개가 모두 일치하면 확률 1 / 2개 1개로 나뉘면 확률 0.66, 0.33

knn1.score(te_x, te_y)
# 이웃 점 숫자 설정에 따라 score가 달라짐
# 점 3개 비교 0.825



from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors = 2)

knn1.fit(tr_x, tr_y)
pred5 = knn1.predict(te_x)
pred5 = knn1.predict_proba(te_x)

knn1.score(te_x, te_y)
# 점 2개 비교 0.809




##########################################################################
# Decision Tree
# 두 개 확률 곱의 숫자가 작을 수록 순수도가 높음 (지니계수)
from sklearn.tree import DecisionTreeClassifier
dt1 = DecisionTreeClassifier()
dt1.fit(tr_x, tr_y)
pred6 = dt1.predict(te_x)
dt1.score(te_x, te_y)
# graphviz windows

from sklearn.tree import export_graphviz
export_graphviz(dt1, "tree1.dot")


from sklearn.tree import DecisionTreeClassifier
dt1 = DecisionTreeClassifier(max_depth = 3)
dt1.fit(tr_x, tr_y)
pred6 = dt1.predict(te_x)
dt1.score(te_x, te_y)

from sklearn.tree import export_graphviz
export_graphviz(dt1, "tree2.dot")

# 랜덤포레스트
# 표본을 모두 다르게 추출해서 Decision Tree 여러 번 실행하고 우세한 결과를 얻음
# 정확성이 아주 높음. 그러나 샘플 수가 많아야 함.

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
rf1.fit(tr_x, tr_y)
rf1.score(te_x, te_y)

rf2 = RandomForestClassifier(n_estimators = 50)
rf2.fit(tr_x, tr_y)
rf2.score(te_x, te_y)
rf2.estimators_
# RandomForest는 Decision Tree를 Estimator로 사용




##########################################################################
# 군집분석
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3)
km.fit_predict(tr_x)





















