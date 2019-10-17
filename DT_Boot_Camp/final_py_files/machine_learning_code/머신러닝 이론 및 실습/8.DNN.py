# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:37:44 2019

@author: pns_com2
"""

#--------------- Data Load --------------#
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import normalize
from imblearn import over_sampling # SMOTE 사용하기 위해서 필요함
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('MLdata2_R.csv', delimiter=',', header=0)
# 8451 rows × 60 columns

#--------------- Data Cleaning --------------#
data_cleaning = data.dropna(axis=0)
# 5196 rows × 60 columns
np.random.seed(12050163)

#--------------- Data Normalization ---------#
data_cleaning = data_cleaning.values  # Numpy representation of NDFrame, an N-dimensional version of a pandas' DataFrame 
data_age = normalize(data_cleaning[:,[1]], axis=0)  # sklearn.preprocessing.normalize,  0: normalize each feature
data_con = normalize(data_cleaning[:,14:41], axis=0)
data_bmi = normalize(data_cleaning[:,[53]], axis=0)
data_cleaning_nor = np.concatenate((data_cleaning[:,[0]],data_age,data_cleaning[:,2:14], data_con,data_cleaning[:,41:53],
                                   data_bmi,data_cleaning[:,54:60]),axis=1)  # Join a sequence of arrays along an existing axis
data_nor = pd.DataFrame(data_cleaning_nor, columns=[data.columns])

data_nor.event

#-------- Prepare Training Data, Test Data ---------#
np.random.seed(12050163)

event_0 = data_nor[np.array(data_nor.event==0)]
event_1 = data_nor[np.array(data_nor.event==1)]

event_0_shuffle = event_0.sample(frac=1)  # DataFrame.sample:Returns a random sample of items from an axis of object, 3580 rows x 60 columns
event_1_shuffle = event_1.sample(frac=1)  # 1616 rows x 60 columns

event_0_training_n = int(len(event_0) * 2/3)
event_1_training_n = int(len(event_1) * 2/3)

training = [event_0_shuffle[:event_0_training_n], event_1_shuffle[:event_1_training_n]]
test = [event_0_shuffle[event_0_training_n:], event_1_shuffle[event_1_training_n:]]
training_data = pd.concat(training)
test_data = pd.concat(test)

np.random.seed(12050163)
training_data_shuffle = training_data.sample(frac=1)
test_data_shuffle = test_data.sample(frac=1)


trainX = training_data.drop('event',1)
# trainX.head()
trainY = training_data[['event']]
# trainY.head()
#SMOTE
#sm = over_sampling.SMOTE(ratio='auto', kind='regular')
#X, y = sm.fit_sample(X, y)
testX = test_data.drop('event',1)
testY = test_data[['event']]

X_train = trainX.values
y_train = trainY.values
X_test = testX.values
y_test = testY.values


#-------- Single-step GridSearch on the hidden-layer size of Multilayer Neural Network -------------#
# Training MLP
mlp = MLPClassifier()
param_grid = {'hidden_layer_sizes': [i for i in range(4,25)],
              'activation': ['tanh'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [10000],
              'early_stopping': [False],
              'warm_start': [False]}

GS_CV = GridSearchCV(mlp, param_grid)
GS_CV.fit(X_train, y_train)
GS_CV.best_params_    # 'hidden_layer_sizes': 19

# Training MLP
mlp_clf = MLPClassifier(hidden_layer_sizes=(19))
mlp_clf.fit(X_train,y_train)

# Prediction of MLP
predict_y = mlp_clf.predict(X_test)

acc = accuracy_score(predict_y, y_test)
roc = roc_auc_score(y_test, mlp_clf.predict_proba(X_test)[:,1])

print("Accuracy Score:   ", acc)
print("ROC AUC Score:   ", roc)

from sklearn.metrics import classification_report

# Confusion matrix
print(confusion_matrix(y_test,predict_y)) 

# Precision, Recall, F1-score   
print(classification_report(y_test,predict_y))




