# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:28:52 2019

@author: HP27
"""
import pandas as pd

data1=pd.read_csv("./data/서울시 공공자전거 대여소 정보_20181129.csv")
gu2018.["대여일자"].apply()
"'abcd'".replace("'","")

import os
import os.path as op
op.curdir.
os.chdir('./data')
list1=os.listdir()
list1[0]
base=pd.read_csv(list1[0], encoding="utf-8")
# (1) 파일명
# (2) 파일 내용에 한글포함 encoding="utf-8"
# (3) 파일 다시 저장하기 -> encoding="utf-8"

globals()
globals()['y']=100
y+5

import pandas as pd
import os
import os.path as op
os.chdir('./data')
list1=os.listdir()
f_list=["base","gu2017","gu2018","user2017","user2018"]

for i in range(len(list1)):
    globals()[f_list[i]]=pd.read_csv(list1[i], encoding="utf-8")

gu2017.columns

import pandas as pd # read_csv
import os  # os.listdir()
os.curdir
list1=os.listdir()
list1[0]

base=pd.read_csv(list1[0], encoding="utf-8")
gu2017=pd.read_csv(list1[1], encoding="utf-8")
gu2018=pd.read_csv(list1[2], encoding="utf-8")
user2017=pd.read_csv(list1[3], encoding="utf-8")
user2018=pd.read_csv(list1[4], encoding="utf-8")

base.columns
gu2017.columns
gu2018.columns
gu=pd.concat([gu2017.copy(), gu2018])
gu2018.shape
gu.shape
user2017.columns.values == user2018.columns.values
user=pd.concat([user2017.copy(), user2018])

type(user2017.columns)
user2017.columns == "'대여소번호'"

user.columns.values[0]="대여일자"
gu.columns.values[0]="대여일자"

gu.columns
user.columns
#(1) 2017년 4월 vs 2018년 4월 활성여부 확인
#(2) 2017년 4월 vs 2018년 4월 이동거리 변화 확인
#(3) 행정구역별로 2018년 4월 데이터에 대해서 유사한 행정구역 찾기
gu.columns
gu.head()
y2017=gu[gu['대여일자']=="'201704'"][["'대여소번호'","'대여건수'"]]
y2018=gu[gu['대여일자']=="'201804'"][["'대여소번호'","'대여건수'"]]

gu['대여일자'].value_counts()
y="abc"
y
gu2017["대여일자"].value_counts()
gu2017[gu2017['대여일자']==201704]
############################################
gu.dtypes


y2017=gu[gu['대여일자']=="'201704'"][["'대여소번호'","'대여건수'"]]
y2018=gu[gu['대여일자']=="'201804'"][["'대여소번호'","'대여건수'"]]


##############################################

y2017.dtypes
y2018.dtypes

Q1=pd.merge(y2017, y2018, on="'대여소번호'")
Q1.shape
y2018.shape

from scipy.stats import ttest_rel
ttest_rel(Q1.iloc[:,1], Q1.iloc[:,2])

대여일자
대여일자'
'대여일자'
# replace(찾기, 바꾸기)
"'abcd'".replace("'","")
gu2018.columns
a2=gu2018["'대여소번호'"]
type(a2[0])
gu2018.dtypes=="object"
type(a2)
a2.dtype
a2.describe()
type(gu2017)
type(a2)
gu2018.applymap(lambda x: str(x).replace("'",""))
# str(값) : 값을 문자열로 변환
str(3).replace()
3

#(2) 2017년 4월 vs 2018년 4월 이동거리 변화 확인
#(3) 행정구역별로 2018년 4월 데이터에 대해서 유사한 행정구역 찾기
gu.columns.values
['대여일자', "'대여소번호'", "'대여소'", "'대여건수'", "'반납건수'"]
user.columns.values
['대여일자', "'대여소번호'", "'대여소'", "'대여구분코드'", "'성별'", "'연령대코드'",
       "'이용건수'", "'운동량'", "'탄소량'", "'이동거리(M)'", "'이동시간(분)'"]

user2017=pd.read_csv(list1[3], encoding="utf-8")
user2018=pd.read_csv(list1[4], encoding="utf-8")
user=pd.concat([user2017, user2018])
user.columns
user["'대여일자'"].value_counts()
date_list=["'201704'","'201804'"]
cond1=user["'대여일자'"].isin(date_list) 
gp_list=["'대여일자'","'대여소번호'"]
Q2=user[cond1].groupby(gp_list)["'이동거리(M)'"].mean()
Q2.shape
Q2_1=Q2.index.to_frame(index=True)
Q2_1.head()
Q2_2=pd.concat([Q2_1, Q2], axis=1)
Q2_2.head()
Q2_2.columns
['대여일자', ''대여소번호'', ''이동거리(M)'']
Q23=Q2_2[Q2_2["'대여일자'"]=="'201704'"]
Q24=Q2_2[Q2_2["'대여일자'"]=="'201804'"]
Q2_3.columns
Q2_4.columns

Q234=Q2_2[Q2_2["'대여일자'"].isin(date_list)]
Q243=Q234.unstack(level=0)
Q244=Q243.dropna()

# Q2_f=pd.merge(Q2_3, Q2_4, on="'대여소번호'")

Q2_f=pd.pivot_table(user[user["'대여일자'"].isin(date_list)],
                    index="'대여소번호'",
                    columns="'대여일자'",
                    values="'이동거리(M)'")

Q2_f2=Q2_f.dropna()
type(Q2_f2)
Q2_f2.shape
Q2_f2.columns
ttest_rel(Q2_f2.iloc[:,0], Q2_f2.iloc[:,1])
Q2_f2.iloc[0:2,"'201704'"]
Q2_f2.loc[:,0]













