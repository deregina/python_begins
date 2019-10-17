# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:38:01 2019

@author: user
"""

# 폴더가 import 우선순위, 
# working directory에 pandas라는 폴더가 있으면 폴더가 import됨
import pandas as pd
# 현재 폴더의 하위에 있는 경우 . --> 현재 표시
data1 = pd.read_csv("./data/서울시 공공자전거 대여소 정보_20181129.csv")

import os
import os.path as op
op.curdir

# 원하는 위치로 이동해서 파일명 list 만들기
os.curdir
os.chdir('./data')
os.listdir()
list1 = os.listdir()
f_list = ['base', 'gu2017', 'gu2018', 'user2017', 'user2018']

# (1) 파일명에 한글 없애기
# (2) 파일 내용에 한글 포함된 경우 encoding = "utf-8"로 시도
# (3) 그래도 안되면 data를 메모장으로 열어서 
#     모든 파일 / uft-8로 설정, 다른 이름으로 저장

base = pd.read_csv(list1[0], encoding = "utf-8")

# 문자열을 변수로 바꾸기
# globals()["문자열"] = value 

for i in range (len(list1)) :
    globals()[f_list[i]] = pd.read_csv(list1[i], encoding = "utf-8")

base.columns
gu2017.columns
gu2018.columns
# copy 안하면 나중에 앞 쪽 DataFrame도 함께 수정됨
# 뒤 DataFrame은 어차피 복사됨
# ....? 다 바뀌는데?
gu = pd.concat([gu2017.copy(), gu2018])
# 혹은 : gu = pd.concat([gu2017, gu2018], copy = True)
gu2018.shape
gu2017.shape
gu.shape
user2017.columns
user2018.columns
user2017.columns == "대여소번호"
user2017.columns == user2018.columns

# user2017.columns.values는 columns 이름을 array로 줌
# 그래서 하나하나 비교가 가능!
# index는 참조만 가능하기 때문에 수정이 불가
user2017.columns.values == user2018.columns.values
user = pd.concat([user2017.copy(), user2018])
# 혹은 : user = pd.concat([user2017, user2018], copy = True)
user.columns.values[0] = "대여일자"
gu.columns.values[0] = "대여일자"

# << 분석목표 >>
# 1. 2017년 4월에 비해 2018년 4월에 사용이 활성화 되었는가?
# 2. 2017년 4월과 2018년 4월 이동거리 비교 
# 3. 행정구역별로 2018년 4월 데이터에 대해서 유사한 행정구역은 어디?

gu.columns
gu.head()

# 대여일자가 201704에 해당하는 행들에서만 대여건수를 가져옴
# 날짜 int로...
gu.dtypes
y2017 = gu[gu['대여일자'] == 201704][['대여소번호', '대여건수']]
y2017
y2018 = gu[gu['대여일자'] == 201804][['대여소번호', '대여건수']]
y2018

# 201704에 해당하는 value 찾아보면 487개 동일!
gu['대여건수'].value_counts()
y = "abc"
y2017.dtypes
y2018.dtypes

y2017['대여소번호'].values == y2018['대여소번호'].values
Q1 = pd.merge(y2017, y2018, on = '대여소번호')
Q1.shape
y2018.shape
y2017.shape


from scipy.stats import ttest_rel
# 2017년과 2018년을 같다고 상정한 T(정규분포)
# pvalue가 5%보다 작으면 
# statistic이 음수면 2017 < 2018 (2017-2018)
ttest_rel(Q1.iloc[:, 1], Q1.iloc[:, 2])
ttest_rel(Q1["대여건수_x"], Q1["대여건수_y"])

# << 분석목표 >>
# 1. 2017년 4월에 비해 2018년 4월에 사용이 활성화 되었는가? O




# replace(찾기, 바꾸기)
"'abcd'".replace("'", "")
gu2018.columns
a2= gu2018["대여소번호"]
gu2018.dtypes == "object"
type(a2)
a2.dtype
# Series는 apply 사용
a2.apply(lambda x: str(x).replace("'", ""))
a2= gu2018[["대여소번호"]]
# 
gu2018.applymap(lambda x: str(x).replace("'",""))


# 2. 2017년 4월과 2018년 4월 이동거리 비교 

user.columns.values
cond1 = user["대여일자"].isin([201704, 201804])
cond1
Q2 = user[cond1].groupby(["대여일자", "대여소번호"])["이동거리(M)"].mean()
user[cond1]
Q2.shape
Q2_1 = Q2.index.to_frame()
Q2_1.head()
Q3 = pd.concat([Q2_1, Q2], axis = 1)
Q3.head()

pd.merge(Q3[Q3["대여일자"] == 201704], Q3[Q3["대여일자"] == 201708], 
         on = "대여소번호", left_index = False, right_index = False)

Q2f = pd.pivot_table(user[user['대여일자'].isin([201704, 201804])],
                    index = '대여소번호',
                    columns = '대여일자',
                    values = '이동거리(M)')
# ttest_rel을 하기 위해서는 이런 식으로 데이터 정리를 해야 함

Q2f2 = Q2f.dropna()
Q2f2.shape
Q2f.shape









# 3. 행정구역별로 2018년 4월 데이터에 대해서 유사한 행정구역은 어디?























