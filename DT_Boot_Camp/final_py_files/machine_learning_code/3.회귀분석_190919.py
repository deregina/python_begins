# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:06:45 2019
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')


df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df.head()

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()

# 방금 그래프로 확인했던 변수 간의 상관관계 실제 상관계수로 확인하기
# 절댓값 숫자가 큰 것이 상관관계가 높음
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)


# 상관계수를 색깔의 진하기로 표시하는 heatmap
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

from statsmodels.formula.api import ols

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()

sns.reset_orig()

lm_model = ols('MEDV~RM', df).fit()
# ols(회귀분석 함수) 안의 값들을 자세히 보기
lm_model.params
lm_model.summary()




lm_model._wrap_attrs
lm_model.pvalues
lm_model.fittedvalues

anova_lm(lm_model)
# import statsmodels.api as sm
# sm.stats.anova_lm(lm_model)


# '~' 안쓰는 형태 --> lm.OLS(self, endog ...)
import statsmodels.regression.linear_model as lm

xx = df.drop(columns = "MEDV")
yy = df.MEDV
lm_model2 = lm.OLS(yy, xx).fit()
lm_model2.pvalues
lm_model2.summary()

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                   MEDV   R-squared (uncentered):                   0.959
# Model:                            OLS   Adj. R-squared (uncentered):              0.958
# Method:                 Least Squares   F-statistic:                              891.3
# Date:                Thu, 19 Sep 2019   Prob (F-statistic):                        0.00
# Time:                        15:57:48   Log-Likelihood:                         -1523.8
# No. Observations:                 506   AIC:                                      3074.
# Df Residuals:                     493   BIC:                                      3128.
# Df Model:                          13                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# CRIM          -0.0929      0.034     -2.699      0.007      -0.161      -0.025
# ZN             0.0487      0.014      3.382      0.001       0.020       0.077
# INDUS         -0.0041      0.064     -0.063      0.950      -0.131       0.123
# CHAS           2.8540      0.904      3.157      0.002       1.078       4.630
# NOX           -2.8684      3.359     -0.854      0.394      -9.468       3.731
# RM             5.9281      0.309     19.178      0.000       5.321       6.535
# AGE           -0.0073      0.014     -0.526      0.599      -0.034       0.020
# DIS           -0.9685      0.196     -4.951      0.000      -1.353      -0.584
# RAD            0.1712      0.067      2.564      0.011       0.040       0.302
# TAX           -0.0094      0.004     -2.395      0.017      -0.017      -0.002
# PTRATIO       -0.3922      0.110     -3.570      0.000      -0.608      -0.176
# B              0.0149      0.003      5.528      0.000       0.010       0.020
# LSTAT         -0.4163      0.051     -8.197      0.000      -0.516      -0.317
# ==============================================================================
# Omnibus:                      204.082   Durbin-Watson:                   0.999
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1374.225
# Skew:                           1.609   Prob(JB):                    3.90e-299
# Kurtosis:                      10.404   Cond. No.                     8.50e+03
# ==============================================================================

# AIC가 줄어듦
xx1 = xx.drop(columns = "INDUS")
lm_model3 = lm.OLS(yy, xx1).fit()
lm_model3.summary()

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                   MEDV   R-squared (uncentered):                   0.959
# Model:                            OLS   Adj. R-squared (uncentered):              0.958
# Method:                 Least Squares   F-statistic:                              967.5
# Date:                Thu, 19 Sep 2019   Prob (F-statistic):                        0.00
# Time:                        15:59:01   Log-Likelihood:                         -1523.8
# No. Observations:                 506   AIC:                                      3072.
# Df Residuals:                     494   BIC:                                      3122.
# Df Model:                          12                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# CRIM          -0.0928      0.034     -2.701      0.007      -0.160      -0.025
# ZN             0.0488      0.014      3.412      0.001       0.021       0.077
# CHAS           2.8482      0.898      3.171      0.002       1.083       4.613
# NOX           -2.9275      3.222     -0.909      0.364      -9.258       3.403
# RM             5.9318      0.303     19.555      0.000       5.336       6.528
# AGE           -0.0073      0.014     -0.527      0.598      -0.034       0.020
# DIS           -0.9655      0.189     -5.099      0.000      -1.337      -0.593
# RAD            0.1723      0.064      2.687      0.007       0.046       0.298
# TAX           -0.0095      0.004     -2.693      0.007      -0.016      -0.003
# PTRATIO       -0.3930      0.109     -3.607      0.000      -0.607      -0.179
# B              0.0149      0.003      5.544      0.000       0.010       0.020
# LSTAT         -0.4165      0.051     -8.225      0.000      -0.516      -0.317
# ==============================================================================
# Omnibus:                      204.123   Durbin-Watson:                   0.999
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1374.966
# Skew:                           1.609   Prob(JB):                    2.69e-299
# Kurtosis:                      10.406   Cond. No.                     8.16e+03
# ==============================================================================

xx2 = xx.drop(columns = "AGE")
lm_model4 = lm.OLS(yy, xx1).fit()
lm_model4.summary()

#                                  OLS Regression Results                                
# =======================================================================================
# Dep. Variable:                   MEDV   R-squared (uncentered):                   0.959
# Model:                            OLS   Adj. R-squared (uncentered):              0.958
# Method:                 Least Squares   F-statistic:                              967.5
# Date:                Thu, 19 Sep 2019   Prob (F-statistic):                        0.00
# Time:                        16:00:39   Log-Likelihood:                         -1523.8
# No. Observations:                 506   AIC:                                      3072.
# Df Residuals:                     494   BIC:                                      3122.
# Df Model:                          12                                                  
# Covariance Type:            nonrobust                                                  
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# CRIM          -0.0928      0.034     -2.701      0.007      -0.160      -0.025
# ZN             0.0488      0.014      3.412      0.001       0.021       0.077
# CHAS           2.8482      0.898      3.171      0.002       1.083       4.613
# NOX           -2.9275      3.222     -0.909      0.364      -9.258       3.403
# RM             5.9318      0.303     19.555      0.000       5.336       6.528
# AGE           -0.0073      0.014     -0.527      0.598      -0.034       0.020
# DIS           -0.9655      0.189     -5.099      0.000      -1.337      -0.593
# RAD            0.1723      0.064      2.687      0.007       0.046       0.298
# TAX           -0.0095      0.004     -2.693      0.007      -0.016      -0.003
# PTRATIO       -0.3930      0.109     -3.607      0.000      -0.607      -0.179
# B              0.0149      0.003      5.544      0.000       0.010       0.020
# LSTAT         -0.4165      0.051     -8.225      0.000      -0.516      -0.317
# ==============================================================================
# Omnibus:                      204.123   Durbin-Watson:                   0.999
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1374.966
# Skew:                           1.609   Prob(JB):                    2.69e-299
# Kurtosis:                      10.406   Cond. No.                     8.16e+03
# ==============================================================================
# AIC 값이 최대한 작아지는 방향으로 변수를 선택해야 함
# 상관없는 변수를 없애면 AIC가 줄어듦

X = df[['RM']].values
y = df['MEDV'].values

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

