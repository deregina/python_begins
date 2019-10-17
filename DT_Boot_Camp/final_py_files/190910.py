# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:43:08 2019

@author: user
"""

# 자료 종류, 자료구조
# 리스트, 튜플, 딕셔너리
# [], (), {"키": 값, "키": 값}
# list(), tuple(), dict("키": 값, "키": 값)
# 상수 단위 처리 x0 = 3
# 접근 : 
# - 숫자접근 [index 방식] 변수명[번호(0 ~)], x1[0], x1[0][1]
# - 키 방식 : 변수명["키"] x2["Age"], x2["Age"][2]

x1 = [[[[[1, 2], 3], 3], 4, [5, 8]], [5, 7, [1, 2]], 9]
len(x1)
x1[1]           # [5, 7, [1, 2]]
x1[1][2]        # [1, 2]
x1[1][2][0]     # 1
x1.append(5)
x1
x2 = (3, 5, 2, (5, 2, 5, (7, 4)))
len(x2)         # 4
type(x2)        # tuple, append 사용 불가
x2[3]           # (5, 2, 5, (7, 4))
x2[3][3]        # (7, 4)
x2[3][3][0]     # 7
type(x2[3][3])  # tuple
x3 = {"age" : [3, 5], "name" : ["Kim", "Lee", "Ha"]}
type(x3)
x3["age"]
x3["age"][0]
x3.keys()
x3.values()
x3["height"] = [170, 163, 183]
x3
# x3 = {"age" : [3, 5], 
#       "name" : ["Kim", "Lee", "Ha"], 
#       "height" : [170, 163, 183]}

x0 = {"weight" : [15, 20, 68]}
x3.update(x0)
x3
x3.update(age = [10, 9, 8])
# x3.update(이미 존재하는 키 = 바꾸고 싶은 값)

x3

x1 > 5          # Error
x1[0] > 5       # Error
x1[0][0] > 5    # 상수 단위만 처리 가능

x4 = [5, 3, 2, 7, 6, 9, 2]
x4 > 5

for i in x4 :
    print(i)
# 요소 프린트
    
for i in range (len(x4)) :
    print(x4[i])
# 요소 프린트
    
for i in range (len(x4)) :
    print(i)
# 0~6 프린트 
    
for i in range (len(x4)) :
    print(x4[i] > 5)
print(i)
# 들여쓰기가 들어간 곳까지 한 문장
# 들여쓰기 들어간 곳 작업이 끝나야 넘어갈 수 있음
# False False False True True True False 6

for i in range (len(x4)) :      # [0, 1, 2, 3, 4, 5, 6]
    print(x4[i] > 5)
    print(i)
# 들여쓰기가 있으므로 모두 한 문장
# False 0 False 1 False 2 True 3 True 4 True 5 False 6
    

range(1, 5)                     # 1, 2, 3, 4  출력은 안됨
for i in range(1, 5):
    print(i)
    
np.arange(1, 5)                 # array([1, 2, 3, 4]) 출력
np.arange(5)
np.arange(0, 10, 3)
    
print(np.arange(len(x4)))       # [0 1 2 3 4 5 6]
np.arange(len(x4))              # array([0, 1, 2, 3, 4, 5, 6])

for i in np.arange(len(x4)):
    print(i)

import numpy as np
for i in np.arange(len(x4)) :   # [0, 1, 2, 3, 4, 5, 6]
    print(x4[i] > 5)


#  중첩 for loop
for i in x1 :
    for j in i :
        for k in j :
            print(k)
            
#  x1 때문에 실행은 끝까지 안된다
for i in np.arange(len(x1)) :
    for j in np.arange(len(x1[i])) :
        for k in np.arange(len(x1[i][j])) :
            print(x1[i][j][k])
            
# in range로 하는 거랑 차이가 없지만 
# 출력해서 확인해 보려면 np.arange로 해보는 게 낫다    
x5 = [[1, 2, 3], [4, 5, 6]]
for i in np.arange(len(x5)):
    for j in np.arange(len(x5[i])) :
        print(x5[i][j])
        
# 들여쓰기 끝나는 부분 주의
for i in x5 :
    for j in i :
        print(j, end = "\t")    
        
for i in x5 :
    for j in i :
        print(j, end = "\t")
        print()
        
for i in x5 :
    for j in i :
        print(j, end = "\t")
    print()
    
# if 조건문
# 조건이 참인 경우 명령 실행
# continue : 밑에 실행 안하고 skip
for i in x5 :
    for j in i :
        if j == 5 : continue
        print(j, end = "\t")  
        
# break : loop 종료
for i in x5 :
    for j in i :
        if j == 5 : break
        print(j, end = "\t")  
  
# *를 변수 앞에 쓰면 갯수를 안 맞춰도
# 나머지 요소들이 다 그 변수에 들어감        
a1, a2 = 3, 4
a3 = 3, 4
a4, a5 = 3, 4, 7 
a7, *a8, a9 = 3, 4, 7, 9, 10
print(a4)
print(a5)
print(a7)
print(a8)
print(a9)


# Unpacking   분리해서 받기
family = {"dad" : "homer", "mom" : "marge", "daughter" : "Lisa"}
for key, value in family.items() :
    print(key, value)
    
for i, j in x3.items() :
    print(i, j)
    print(j, "\n")
    
for i in x3.items() :
    print(i)
    
# numpy.org
# pypi.org에서 Keyword 검색


############################### pandas ###############################

# !pip install 패키지명
# !conda 패키지명

# !cd
# !pip install pandas

# Package - Module 
# uci machine learning repository
# seperator 기본값이 ','니까 안 써도 됨
# Header 행이 없는 경우 "header = 0", "header = None"을 써서
# 첫번째 행을 header로 잡지 않도록 설정해줘야 함
import pandas as pd
data1 = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", sep=",")
data2 = pd.read_csv("breast-cancer-wisconsin.data", header = None)

type(data2)
# data2.head()는 5행만 가져옴
data2.head()
# data2.describe() data 분석 통계들을 보여줌
data2.describe()
# data2[1] 1열을 가져옴 
data2[1]
# data2[1].plot() 차트를 그려줌
data2[1].plot()
data2[1] > 5
# [조건]에 해당하는 data 행만 뽑아줌
data2[data2[1] > 5]


!pip install matplotlib
import matplotlib.pyplot as plt
data2[1].plot()

# pd.read_csv("파일명", na_values = "", header = 0, sep = "\t",  )
# "C:/폴더명/폴더명/파일명.csv"
# "C:\\폴더명\\폴더명\\파일명.csv"

data2.columns.values    # Series 형태로 줌
data2.dtypes

# Series가 모여서 DataFrame, DataFrame의 각 줄은 Series
type(data2[1])          # DataFrame
data2[1][3]             # [열][행]
data2[2,3]              # Error
 data2.iloc[3, 1]       # [행, 열]

## pandas methods
# as_"type"             형 변화
# bfill                 빈자리 채우기
# cummax                변수 간의 상관관계
# count                 n/a 아닌 것 개수 count
# cov
# drop XX
# duplicated            반복되는 것 찾기
# fill XX
# is XX                 checking
# pivot
# sort                  행 단위로 정렬

# pandas는 기본적으로 열 단위로 계산한다
data2.describe()
# 열별 산술평균 구하기
data2.mean()
# 중간값 구하기, 값이 짝수개일 경우 가운데 두 값의 평균을 줌
data2.median()
# mean과 median의 차이가 많이 나면 이상치가 있다고 의심할 수 있음
# 평균이 median보다 훨씬 크면 큰 쪽에 이상치가 있을 가능성이 큼
# 분산 : 얼만큼 떨어져 있나, 변동폭
# 정규분포 : 보아뱀 모양

# 한 그룹의 분산이 작다 = 동질적이다 = 분석하기 좋다
# 그룹 사이의 간격은 넓을수록 좋다 
# 분산이 0이면 그 attribute는 모두 같은 값 => y와 상관관계가 없다 => 버림

# Variation 분산
data2.var()
# Standard Deviation 표준편차
data2.std()

# 분위수 : 1/4, 2/4, 3/4 지점
data2.quantile()
data2.quantile(0.25)
data2.quantile(0.75)
# 상위 5%, 25%, 75%, 95%
data2.quantile([0.05, 0.25, 0.75, 0.95])
data2.boxplot()
# Series는 boxplot 선택 불가
# 25%, 75% 사이에 박스가 그려지고 중앙 값인 50에 녹색선
# IQR; inter-quartile range = Q3 - Q1
# 25%, 75%의 
# 박스가 작을 수록 동질성이 높음
# 1.5 IQR을 벗어난 데이터, 즉 바깥쪽 울타리에 있는 데이터는 
# 더욱 특수한 데이터이므로 개별 데이터를 모두 점으로 찍어서 나타낸다
data2[0].boxplot()
data2[1:3].boxplot()
data2[1:2].plot()
data2.boxplot(column = [1, 2]) 

# 공분산 : 공분산(covariance)은 2개의 확률변수의 상관정도를 나타내는 값
# cov = E((X-X평균)(Y-Y평균))
import numpy as np
np.mean((data2["A"] - data2["A"].mean())*(data2["B"] - data2["B"].mean()))

data2.columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
data2[["B", "C", "D"]].cov()
# 열을 번호로 연속 지정하고 싶으면 반드시 iloc을 사용
data2.iloc[:, 1:4].cov()
# corr(상관계수, correlation)이 0.2 넘으면 약간의 연관성, 
# 0.6 넘으면 통계적으로 관계성이 높음
data2.iloc[:, 1:4].corr()

data2[["B", "C", "D"][2]]

# Data Frame을 만들 때는 Dictionary 형태로 넣어야만 열 단위로 값이 들어감
dd1 = dict(A = [1, 2, 3])
d1 = pd.DataFrame(dd1)

d2 = pd.DataFrame(dict(A = [1, 3, 5],
                       B = ["a", "b", "d"]))

# dict을 쓰지 않으면 횡방향, List 형태로 값을 하나씩 넣어줌
# .colums 으로 열 이름 지정
# DataFrame 구조에서만 .열이름 으로 값에 접근 가능
d4 = pd.DataFrame([[3, "m", 70], [7, "F", 80]])
d4.columns = ["Age", "Sex", "avg"]
d4["Age"]
d4.Age
type(d4)

# .append는 좌우로만 합치는 것이 가능(열 추가만 가능), 
# 행 단위로 추가하려면 열 이름이 같은 DataFrame을 만들어서 붙여야 함
# 결과를 출력해주지만 실제 DataFrame에는 적용되지 않음
# .append([]) 가장 우측에 새 열을 생성해서 순서대로 모두 들어감
# 각각의 요소가 하나의 행
d4.append([6, "M", "NA"])
# 기존 값을 무시하고 새로운 열 / 행 생성해서 넣음
# [[]] 이중 대괄호를 만들면 요소들이 하나의 행에 들어감 
d4.append([[6, "M", "NA"]], ignore_index = True)
# d4[열 이름] = [아이템]  새로운 열 생성해서 값 넣기
# d4["새로운 열"]  새로운 열을 생성해서 넣을 때는 열 이름을 제공해 주어야 함
# 실제 함수에 추가됨
d4["Heigth"] = [6, "NA"]
d4
# d4.열 이름  형태는 생성이 불가
d4.Hee = [8, 2]

# append는 주로 새로운 DataFrame을 만들어서 붙여넣을 때 사용
# pd.DataFrame([[값 1열], [값 2열]], columns = [열 이름])
# 값 개수가 동일해야 함
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)

d4.append(pd.DataFrame([[6, "M", 50]]))

d4 = d4.drop(columns = "Age")
d4

pd.concat()

data2.columns
list1 = ["A", "C", "E"]
list2 = ["B", "F", "J"]
list3 = ["A", "F", "J"]

s1 = data2[list1]
s2 = data2[list2]
s3 = data2.loc[0:100, list1]
s4 = data2.loc[101:150, list1]
s5 = data2.loc[160:190, list1]
s6 = data2.loc[1:120, list3]

s1.columns
s2.columns

pd.concat([s1, s2])

s1.columns;s2.columns
con1 = pd.concat([s1, s2], axis = 1)
con1.shape
con2 = pd.concat([s3, s4, s5])
s1.shape
s3.shape
s1.head()
con1.head()
con2.iloc[99:104, :]
s3.tail()

# how = "left" left join : 왼쪽을 기준으로 겹치는 자료만 붙임
# how = "right" right join : 오른쪽을 기준으로 겹치는 자료만 붙임
# left, right join에서 만약 테이블에 키 값이 같은 데이터가 여러개 있는 
# 경우에는 있을 수 있는 모든 경우의 수를 따져서 조합을 만들어 낸다.
# inner join : 중복되는 것만 남김
merge1 = pd.merge(s1, s6, on = "A")
merge1.columns
merge1.shape
s1.shape
merge2 = pd.merge(s1, s6, 'left', on = "A")
merge2.shape
len(s1.A.unique())
s1.A.unique()
s1.A.value_counts()
s7 = s1.A.drop_duplicates()
s7
s1

data2[data2.C > data2.C.mean()]
data2[data2.C > data2.C.mean()][["B", "C"]]

# []안 에 있는 열을 한 열에 통합, 행 개수는 X2가 됨
s1.columns
melt1 = pd.melt(s1, "A", ["C", "E"])
melt2 = pd.melt(s1, "A", ["C", "E"],
                var_name = "class", 
                value_name = "avg")






















































