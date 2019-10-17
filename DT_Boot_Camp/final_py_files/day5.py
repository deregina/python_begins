# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:10:48 2019

@author: HP27
"""

import numpy as np
import matplotlib.pyplot as plt

x1=np.random.randint(0, 100, 10000)
plt.hist(x1)
m2=[];s3=[]
for i in np.arange(100000):
    x1=np.random.randint(1, 101, 10)
    m1=np.mean(x1);s2=np.var(x1)
    m2.append(m1);s3.append(s2)
        
np.mean(m2);np.mean(np.arange(1,101))
plt.hist(m2);plt.hist(s3)

ss=np.arange(1,101)
np.mean((ss-np.mean(ss))**2)
np.var(m2)















