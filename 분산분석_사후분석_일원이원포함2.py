# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 00:39:39 2019

@author: Mac
"""
##  분산분석 실습 예제 모음

import scipy.stats as stats
import pandas as pd
import urllib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np

## 분산분석을 위한 분산의 특징 확인을 위한 데이터 시각화
centers = [5,5.3,4.5]
std = 0.1
colors = 'brg'

data_1 = []
for i in range(3):
    data_1.append(stats.norm(centers[i], std).rvs(100))
    plt.plot(np.arange(len(data_1[i]))+i*len(data_1[0]),data_1[i], '.', color = colors[i])


std_2 = 2
data_2 = []
for i in range(3):
    data_2.append(stats.norm(centers[i], std_2).rvs(100))
    plt.plot(np.arange(len(data_1[i]))+i*len(data_2[0]), data_2[i], '.', color = colors[i])
    
    

# =============================================================================
# 예시 데이터(Altman 910)
#
# 22명의 심장 우회 수술을 받은 환자를 다음의 3가지 그룹으로 나눔
# 
# Group I: 50% 아산화 질소(nitrous oxide)와 50%의 산소(oxygen) 혼합물을 24시간 동안 흡입한 환자
# Group II: 50% 아산화 질소와 50% 산소 혼합물을 수술 받는 동안만 흡입한 환자
# Group III: 아산화 질소 없이 오직 35-50%의 산소만 24시간동안 처리한 환자
# => 적혈구의 엽산 수치를 24시간 이후에 측정
# =============================================================================

# url로 데이터 얻어오기
url = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# Sort them into groups, according to column 1
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

# matplotlib plotting
plot_data = [group1, group2, group3]
ax = plt.boxplot(plot_data)
plt.show()

# =============================================================================
## Boxplot 특징
# 평균값의 차이가 실제로 의미가 있는 차이인지, 
# 분산이 커서 그런것인지 애매한 상황
# =============================================================================

# =============================================================================
# scipy.stats으로 일원분산분석
# =============================================================================

F_statistic, pVal = stats.f_oneway(group1, group2, group3)

print('Altman 910 데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.5f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미한 차이가 있음')

# =============================================================================
# statsmodel을 사용한 일원분산분석
# =============================================================================

# 경고 메세지 무시하기
import warnings
warnings.filterwarnings('ignore')

df = pd.DataFrame(data, columns=['value', 'treatment'])    

# the "C" indicates categorical data
model = ols('value ~ C(treatment)', df).fit()

print(anova_lm(model))

# =============================================================================
# 이원분산분석(two-way ANOVA)
# => 독립변인의 수가 두 개 이상일 때 집단 간 차이가 유의한지를 검증
# 
# 상호작용효과(Interaction effect)
# => 한 변수의 변화가 결과에 미치는 영향이 다른 변수의 수준에 따라 달라지는지를 확인하기 위해 사용
# 
# 예제 데이터(altman_12_6) 설명
# => 태아의 머리 둘레 측정 데이터
# => 4명의 관측자가 3명의 태아를 대상으로 측정
# => 이를 통해서 초음파로 태아의 머리 둘레측정 데이터가 
#    재현성이 있는지를 조사
# =============================================================================

inFile = 'altman_12_6.txt'
url_base = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/'
url = url_base + inFile
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# dataframe-format로 데이터셋 가져오기
df = pd.DataFrame(data, columns=['head_size', 'fetus', 'observer'])
# df.tail()

# 태아별 머리 둘레 plot
df.boxplot(column = 'head_size', by='fetus' , grid = False)

# =============================================================================
# #그림 결과 설명
# #태아(fetus) 3명의 머리 둘레는 차이가 있어보임
# 이것이 관측자와 상호작용이 있는것인지 분석을 통해 확인 필요
# =============================================================================

# =============================================================================
# 분산분석으로 상호(상관, 교호)관계 파악
# =============================================================================
formula = 'head_size ~ C(fetus) + C(observer) + C(fetus):C(observer)'
lm = ols(formula, df).fit()
print(anova_lm(lm))

# =============================================================================
# 결과 설명
# P-value 가 0.05 이상. 따라서 귀무가설을 기각할 수 없음
#측정자와 태아의 머리둘레값에는 연관성이 없다고 할 수 있음
#  측정하는 사람이 달라도 머리 둘레값은 일정하는 의미
# 
# 결론적으로 초음파로 측정하는 태아의 머리둘레값은 믿을 수 
# 있다는 의미
# =============================================================================

# =============================================================================
# 사후분석(Post Hoc Analysis)
# =============================================================================

import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np

# url로 데이터 얻어오기
url = 'https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# 그룹 단위로 불러오기
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

# pandas로 데이터 불러오기
df = pd.DataFrame(data,columns=['value', 'treatment']).set_index('treatment')

# 예시 데이터 시각화 하기
plot_data = [group1, group2, group3]
ax = plt.boxplot(plot_data)
plt.show()


df.head()

df2 = df.reset_index()
df2.head()

from statsmodels.stats.multicomp import pairwise_tukeyhsd

posthoc = pairwise_tukeyhsd(df2['value'], df2['treatment'], alpha=0.05)
print(posthoc)

fig = posthoc.plot_simultaneous()



