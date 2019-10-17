# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:34:27 2019

@author: user
"""

import scipy.stats as sc
import numpy as np
import pandas as pd

np.random.seed(0)
x = sc.norm(0, 1).rvs(100)
# or --> x1 = np.random.randn(3, 5)
np.mean(x)

# < 단일검정 (ttest_1samp(x, popmean: 귀무가설의 기댓값)) >
sc.ttest_1samp(x, 0)
sc.ttest_1samp(x, 5)
# --> x는 표준정규분포와 같다

sleep = pd.read_csv("sleep.csv")
x_bar = sleep.extra.mean()              # 표본평균
x_std = sleep.extra.std()               # 표본의 표준편차
x_n = len(sleep.extra)                  # 표본의 수
mu = 0
(x_bar - mu)/(x_std/np.sqrt(x_n))       # 검정통계량 (np.sqrt는 루트 역할)
sc.ttest_1samp(sleep["extra"], 0)       # popmean = 0이라는 귀무가설 기각
# (statistic=3.412964995270109, pvalue=0.002917620404154116)
# --> 감기환자의 수면 시간이 증가                   
# H0 : 감기가 걸려도 수면 양에 변화가 없다
# H1 : 감기가 걸리면 수면 양에 변화가 있다

sc.ttest_1samp(sleep["extra"], 0)
sc.ttest_1samp(sleep["extra"][0:10], 0)
sleep.columns
sc.ttest_1samp(sleep[sleep["group"] == 1]["extra"], 0)
# (statistic=1.3257101407138212, pvalue=0.2175977800684489)
# ==> 감기 걸려도 수면 양에는 변화가 없음
sc.ttest_1samp(sleep[sleep["group"] == 2]["extra"], 0)
# (statistic=3.6799158947951884, pvalue=0.005076132649772411)
# ==> 감기 걸린 사람이 약을 먹으면 수면 양에 변화가 있음



###########################################################################

# < 1단계 : 등분산검정 >  F값을 이용
#           H0 = var(a) = var(b)
#           H1 = var(a) != var(b)



# < 독립검정 (ttest_ind) > : 독립인 두 집단 간(a, b)의 비교
#                           두 집단의 평균과 분산이 다르다면 모두 반영해줘야 함
                        # H0 = mu_a = mu_b    (mu_a - mu_b = 0)
                        # H1 = mu_a != mu_b   (mu_a - mu_b != 0)

# 단일검정의 검정통계량 수식
# (x_bar - mu) / (x_std / np.sqrt(x_n))

# 독립검정의 검정통계량 수식
# (a_bar - b_bar) - (mu_a - mu_b) / a_bar와 b_bar를 모두 반영한 표준편차
# = {(a_bar - b_bar) - 0} / sqrt(var(a)/a_n + var(b)/b_n))


x1 = sleep[sleep.group == 1]["extra"]
y1 = sleep[sleep.group == 2]["extra"]
sc.bartlett(x1, y1)
# (statistic=0.10789210747557532, pvalue=0.7425568224059087)
# bartlett test(등분산 여부)를 ttest_ind에 반영해야 함
sc.ttest_ind(x1, y1, equal_var = True)
# (statistic=-1.8608134674868526, pvalue=0.07918671421593818)
# 두 집단은 같다, 수면량에 변화가 없다


# < 대응인 두 집단 간의 차이 검정> : sc.ttest_rel()
# d1 = x1 - y1일 때,
# (d_bar - mu [귀무가설에서 x1_mu = y1_mu]) / (d_std / np.sqrt(d_n))
# = (d_bar - 0) / (d_std / np.sqrt(d_n))
sc.ttest_rel(x1, y1)
# (statistic=-4.062127683382037, pvalue=0.00283289019738427)
# 개인차를 제거하고 두 집단 간의 관계를 봤을 때 
# 두 집단은 같지 않다, 수면량이 늘어났다

rel_data = sleep.groupby(by = ["group", "ID"])["extra"]

# groupby or 아래와 같이 2가지 Index를 가진 DataFrame을 만들 수 있다
# index1 = pd.MultiIndex.from_frame(sleep[["group", "ID"]])
# rel_data = pd.DataFrame(dict(extra = sleep["extra"].values), index = index1)

rel_data1 = rel_data.mean()
rel_data2 = rel_data1.unstack(level = 0)
# 0번째 index를 기준으로(level = 0) unstack (한 줄씩 옆으로, 한 열로 분리)
# rel_data2 = rel_data1.unstack(level = 1)
# --> ID에 10개의 값이 있으므로 10개의 열이 생김

# 보우쌤 보고싶다....

sc.ttest_rel(rel_data2[1], rel_data2[2])

rel_data2["y-x"] = rel_data2[1] - rel_data2[2]
sc.ttest_1samp(rel_data2["y-x"], 0)
# (statistic=-4.062127683382037, pvalue=0.00283289019738427)
# ttest_rel은 두 집단의 개인차를 제거하고 통계량을 계산함
# --> 직접 두 집단을 계산하고 ttest_1samp해도 같은 결과가 나옴


#########################################################################


# < 분산분석 ANOVA > : 세 개 이상의 집단 차이 검정
# 집단들의 분산이 모두 같다는 조건 하에,
# H0 : mu_a = mu_b = mu_c = mu_d = 0
# H1 : 적어도 한 그룹의 mu != 0
# --> 모두 같은지 아니면 한 그룹이라도 다른지 유무만 파악
# 다중비교 후 다르다는 결과가 나오면
# 사후분석 (mu_a = mu_b, mu_a != mu_b, mu_b = mu_c, mu_b = mu_c, ...)
# 을 통해 어느 것이 다른지 알아내야 함


import statsmodels.api as sm
from statsmodels.formula.api import ols
moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load
data = moore.data
data = data.rename(columns={"partner.status" :
                            "partner_status"}) # make name pythonic
moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                data=data).fit()
table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 Anova DataFrame
print(table)

 #                                               sum_sq    df          F    PR(>F)
 # C(fcategory, Sum)                          11.614700   2.0   0.276958  0.759564
 # C(partner_status, Sum)                    212.213778   1.0  10.120692  0.002874
 # C(fcategory, Sum):C(partner_status, Sum)  175.488928   2.0   4.184623  0.022572
 # Residual                                  817.763961  39.0        NaN       NaN

moore_lm1 = ols('conformity ~ fcategory', data = data).fit()
table1 = sm.stats.anova_lm(moore_lm1, typ = 2)
print(table1)

#                 sum_sq    df         F    PR(>F)
# fcategory     3.733333   2.0  0.065037  0.937127
# Residual   1205.466667  42.0       NaN       NaN





















































