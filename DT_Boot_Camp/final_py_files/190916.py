# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:47:22 2019

@author: user
"""

a1 = [3, 6, 9, 2, 7, 20, 50, 6, 7]
# 산술평균(mean) : sum(xi)/n, sum((1/n) * xi) 다 더해서 개수로 나눈 것
#                 각각 비중이 같으므로, 멀리 떨어져 있는 값도 동등하게 영향을 줌 (50)
# 중앙값(median) : 오름차순으로 정렬한 후 정중앙에 위치한 값
#                 홀수인 경우, 짝수인 경우 중앙값 두 개의 평균
# 최빈수(mode) : 가장 자주 나타나는 값 리턴, 같은 횟수로 등장하면 여러 개 리턴
# 기존 통계학에서는 세 가지가 일치할 때 가장 이상적 = 정규분포(평균이 최빈수, 중앙값)
# 산술평균이 중앙값보다 작으면/크면 이상치가 작은 쪽/큰 쪽에 있음

import numpy as np
np.mean(a1)
np.median(a1)
# numpy는 mode 없고, pandas와 scipy에만 있음

import pandas as pd
pd.Series(a1).mean()
a2 = pd.Series(a1)
a2.mean()
a2.median()
a2.mode()

a2.describe()
a2.quantile([0.1, 0.3, 0.9])
# 10%, 30%, 90% 데드라인 찾기
a2.value_counts()

pd.crosstab()
pd.pivot_table()



df = pd.DataFrame({'A':[1, 2, 3, 4, 5],
                   'B':[6, 7, 8, 9, 10],
                   'C':[5, 4, 3, 2, 1]})
pd.crosstab(index = df.A, columns = df.B)

# 1. 깨끗한 데이터를 만들기 위해 전처리
DataFrame.dropna()
DataFrame.fillna(0)
#--> 빈 칸을 0으로 채움

DataFrame.concat()
DataFrame.merge()
DataFrame.isna()
DataFrame.notna()
df.isna()

# N/A일 경우 처리 방법
DataFrame.apply()       # 열 단위
DataFrame.map()         # 셀 단위
DataFrame.applymap()    # 열 + 셀 단위


DataFrame.where(조건(, 실행))
DataFrame[ 조건 ]
groupby, crosstab, pivot_table

# matplotlib
# scipy
# statsmodels


# https://matplotlib.org
# https://python-graph-gallery.com
a2.plot.bar()
type(a2.value_counts())
a2.value_counts().plot.bar()
# 데이터 요약 후 plot.bar 사용
import matplotlib.pyplot as plt
plt.bar(x = np.arange(a2.shape[0]), height = a2)
# x 축에 몇 번째 값인지 표기하게 됨
a2.shape[0]

a3 = a2.value_counts()
type(a3)
a3
a4 = a3.index.values
a4
plt.bar(x = a3.index.values, height = a3)
# x 축에 값 표기
plt.bar(x = a3.index.values, height = a3, color = "red")
plt.barh(width = a3, y = a3.index.values, color = "red")

# array 구조는 numpy를 통해서 mean을 구할 수 있음
b1 = np.array([[3, 5, 8], 
               [5, 7, 2]])
type(b1)
np.mean(b1)
# array 전체를 통째로 인식
np.mean(b1, axis = 0)
np.mean(b1, axis = 1)
b1[0,1]

# 열 / 행 따로따로 인식
b2 = pd.DataFrame(b1)
b2.mean()
np.mean(b2)
np.mean(b2, axis = 1)
b2[0][1]
b2.iloc[1, 0]
# pandas는 기본적으로 열 단위로 처리하므로 plot 사용이 가능
# numpy 형식은 plot 사용 불가능  b1.plot --> 불가


##########################################################################

x = np.arange(1, 10, 0.1)
len(x)
# 간격을 촘촘히 해줘야 부드러운 선이 만들어짐
y = x * 0.2
y2 = np.sin(x)

plt.plot(x, y, 'b', label = 'first', color = "violet")
plt.plot(x, y2, 'r', label = 'second', color = "blue")
plt.title("matplot sample")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc = 'upper right')
plt.show()
# 두 줄 한꺼번에 실행해야 하나의 Figure에 그려줌

# [color][marker][line] or [marker][line][color]
plt.plot(x, y, "go--") 
plt.plot(x, y2, "r:") 
plt.plot(x, y, ":")
 
# 한 함수 안에 여러 Axes 넣을 수 있음
plt.plot(x, y, "gD", x, y2, "r:")



plt.scatter()


#########################################################################
url = "https://raw.githubusercontent.com/duchesnay/pylearn-doc/master/data/salary_table.csv"
df = pd.read_csv(url)
df = salary
# '양'의 상관 (<--> '음'의 상관)

# 색깔을 열 별로 library로 지정
plt.plot(df['experience'], df['salary'], 'o')
edu = salary["education"].unique()
# --> array(['Bachelor', 'Ph.D', 'Master'], dtype=object)
col1 = pd.Series(list(np.where(salary.education == edu[0], 'r',
                      np.where(salary.education == edu[1], 'b', 'k'))))
plt.scatter(df['experience'], df['salary'], marker = 'o', color = col1)

colors = {edu[0] : 'r', edu[1] : 'g', edu[2] : 'k'}
col2 = df['education'].apply(lambda x: colors[x])
plt.scatter(df['experience'], df['salary'], c = col2, s = 50)


########################################################################
# binomial : 이항분포 (Yes or No)

np.random.random([5, 5])

# -4 ~ 4 표준정규분포로 랜덤하게 Size 맞추어 숫자 뽑음
np.random.randn(5, 5)

import scipy.stats as st


#########################################################################

# pmf	확률질량함수(probability mass function) ==> 이산형
# pdf	확률밀도함수(probability density function)
# cdf	누적분포함수(cumulative distribution function) ==> 연속형
# ppf	누적분포함수의 역함수(inverse cumulative distribution function)

# 정규분포 만들기, st.norm()를 쓰면 표준정규분포로 고정됨
n1 = st.norm()
n1.pdf(60)

# cdf 해당 값의 누적 확률을 구해줌
n1.cdf(0)
n1.cdf(0.5)
n1.cdf(1.96) - n1.cdf(-1.96)

# 해당 누적 확률을 가지는 값을 돌려줌
n1.ppf(0.975)
n1.ppf(0.025)


# 평균, 표준편차 입력
n2 = st.norm
n2.cdf(75, 60, 4) - n2.cdf(70, 60, 4)   # 70~75일 확률은 0.612%
(75-60)/4 - (70-60)/4
n2.ppf(0.95, 60, 4) # 66.579 --> 66.58점 넘으면 상위 5%

# 이항분포
n3 = st.binom
# (발생 횟수, 시도 횟수, 확률)
# pmf ( 성공 / 총 시도 수 / 확률)
n3.pmf(1, 5, 1/6)
(1/6 ** 1 + 5/6 ** 5) * 5













































