# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:41:20 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

x1 = np.random.randint(1, 101, 10000)
plt.hist(x1)

m2 = []; s3 = []


for i in np.arange(100000) : 
    x3 = np.random.randint(1, 101, 2)
    m1 = np.mean(x3); s2 = np.var(x3)
    m2.append(m1);    s3.append(s2)

np.mean(m2)
np.mean(np.arange(1, 101))

plt.hist(m2)
plt.hist(s3) 


m2 = []; s3 = []

for i in np.arange(100000) : 
    x3 = np.random.randint(1, 101, 20)
    m1 = np.mean(x3); s2 = np.var(x3)
    m2.append(m1);    s3.append(s2)
    
np.mean(m2)
np.mean(np.arange(1, 101))

plt.hist(m2)
plt.hist(s3)    


# 난수(표본)의 평균과 분산을 10000번 구하니까 
# 평균은 정규분포 형태
# 분산은 카이제곱 분포 형태로 나타남 (뽑는 수가 적어지면 앞으로 이동)
    
ss = np.arange(1, 101)
np.mean((ss - np.mean(ss))**2)
np.var(m2)  
# 모집단의 분산 / 표본의 수 = 표본평균의 분산
# 표본평균은 모집단의 분포보다 당연히 모여있다!


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    