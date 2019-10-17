# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:30:41 2019

@author: user
"""

age = "16 18 20 19 21 22 17 24 23 21 17 17 21 34 56 22 12 15 22 17 44 18"
age_list = age.split(sep = " ")

l2 = list()
for i in range(len(age_list)) :
    l2.append(int(age_list[i]))
l2
len(age_list)
len(l2)

import numpy as np
np.median(l2)
import pandas as pd


l3 = [5, 6, 4, 7, 7, 12, 8]
l4 = pd.DataFrame(l3)
np.mean(l3)
np.median(l3)
l4.mode()

n3.pmf(2, 5, 0.4)

n2 = st.norm
n2.ppf(0.95, 70, 5)

n2.cdf(3.5, 2.8, 0.5) - n2.cdf(3.3, 2.8, 0.5)

df1 = pd.read_csv("sleep.csv")
df2 = df1.iloc[0:10, :]
df3 = df1.iloc[10:20, :]
ttest_rel(df2["extra"], df3["extra"])

ttest_rel(df3["extra"], df2["extra"])
