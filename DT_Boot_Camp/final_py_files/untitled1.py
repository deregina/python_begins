# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:33:29 2019

@author: user
"""

import math
import matplotlib.pyplot as plt

# 정규분포의 밀도 함수
def normal_pdf(x, mu=0, sigma=1):
    return math.exp(((x-mu) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

# 정규분포의 누적분포 함수
# erf : 오차 함수(error function)는 확률론, 통계학, 편미분 방정식등에서 사용하는 비초등 함수이다. 가우스 오차 함수라고도 한다.
#       오차함수는 정규 분포의 누적분포함수와 본질적으로 동일하다.
def normal_cdf(x, mu=0, sigma=1):
    # erf 공식에서는 mu와 sigma는 빠져있다. 여기서는 평행 이동 및 밀도를 보여주기 때문에 추가되었다.
    return (1 + math.erf((x-mu)/math.sqrt(2) / sigma)) / 2


xs = [x/10 for x in range(-50, 50)]

plt.subplot(211)
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-', label='mu=-1, sigma=1')
plt.legend()

plt.subplot(212)
plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-', label='mu=-1, sigma=1')
plt.legend(loc=4)
plt.show()





    
    