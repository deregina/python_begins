# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:00:31 2019

"""
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

training_data = pd.read_csv("training_data.csv", delimiter=',', header=0)
test_data = pd.read_csv("test_data.csv", delimiter=',', header=0)
# 3463 rows X 60 columns
# 1733 rows X 60 columns
X_train = training_data.drop('event',1)  
X_train.head()
#  X_train=X_train.values
y_train = training_data[['event']]  # y_train=y_train.values
X_test = test_data.drop('event',1) # X_test=X_test.values
y_test = test_data[['event']] # y_test=y_test.values

#kNN and NB classification for two-class
#-------- Classificataion Models: kNN and NB -------------#
# ML algorithm
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))

# running algorithms
for name, model in models:
    train_model = model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acc = accuracy_score(prediction, y_test)
    roc = roc_auc_score(prediction, y_test)
    msg = "=============%s=============\n accuracy : %f \n roc_auc: %f" % (name, acc, roc)
    print(msg)

#kNN and NB classification for three-class
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
iris=datasets.load_iris()
X_iris,y_iris=iris.data, iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X_iris,y_iris, test_size=0.3, 
random_state=1)

# running algorithms
for name, model in models:
    train_model = model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acc = accuracy_score(prediction, y_test)
    msg = "=============%s=============\n accuracy : %f "%  (name, acc)
    print(msg)