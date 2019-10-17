# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:32:38 2019

@author: HP27
"""
import scipy.stats as sc
import numpy as np

np.random.seed(0)
# x1=np.random.randn(100)
x=sc.norm(0,1).rvs(100)
x
np.mean(x)

## 단일 검정
sc.ttest_1samp(x, 5)

import pandas as pd
sleep=pd.read_csv("sleep.csv")
x_bar=sleep.extra.mean()
x_std=sleep.extra.std()
x_n=len(sleep.extra)
mu=0
(x_bar-mu)/(x_std/np.sqrt(x_n))
# H0: 감기가 걸리더라도 수면양에는 변화가 없다
# H1: 감기가 걸리면 수면양에는 변화가 있다.
sc.ttest_1samp(sleep.extra, 0)
sc.ttest_1samp(sleep.extra[0:10], 0)
sleep.columns
sc.ttest_1samp(sleep[sleep.group==1]["extra"], 0)
# statistic=1.3257101407138212, 
# pvalue=0.2175977800684489

# ttest_ind : 독립인 두 집단간의 비교
# (1) 전제조건 : 분산이 등산인지 검정
## H0 : var(a) = var(b) : F
## H1 : var(a) != var(b)
## (2) ttest_ind
# =============================================================================
# H0 : mua = mub (mua - mub = 0)
# H1 : mua != mub (mua - mub != 0)
# (x_bar-mu)/(x_std/np.sqrt(x_n))
# 
# 분자:  (a_bar-b_bar)-(0[mua - mub])
#      ------------------------------
# 분모:  sqrt(var(a)/a_n+var(b)/b_n)
# =============================================================================

x1=sleep[sleep.group==1]
y1=sleep[sleep.group==2]

sc.bartlett(x1.extra, y1.extra)
# statistic=0.10789210747557532,
# pvalue=0.7425568224059087
sc.ttest_ind(x1.extra, y1.extra, equal_var=True)

## 대응인 두 집단간의 차이검정
# sc.ttest_rel()
x11-y11=d1
x12-y12=d2
x13-y13=d3
.
.
.x110-y110=d10

rel_data=sleep.groupby(["group","ID"])["extra"]
for i, j in rel_data:
    print(i)
    print(j)

rel_data1=rel_data.mean()
rel_data2=rel_data1.unstack(level=0)

index1=pd.MultiIndex.from_frame(sleep[["group","ID"]])

sleep2=pd.DataFrame(dict(extra=sleep["extra"].values), index=index1)
sleep3=pd.DataFrame(sleep["extra"].values, index=index1)
sleep4=sleep2.unstack(level=0)

sleep5=pd.merge(sleep[sleep.group==1][["extra","ID"]], sleep[sleep.group==2][["extra","ID"]], on="ID")

# H0 : mua = mub (mua - mub = 0)
# H1 : mua != mub (mua - mub != 0)
# (d_bar-0)/(d_std/np.sqrt(d_n))
sc.ttest_rel(y1.extra, x1.extra)
#statistic=-4.062127683382037, 
#pvalue=0.00283289019738427













