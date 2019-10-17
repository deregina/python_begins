# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:00:31 2019
"""
#Decision Tree, Random Forest for two-class 
import pandas as pd
import os
import graphviz

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

training_data = pd.read_csv('training_data.csv', delimiter=',', header=0)
test_data = pd.read_csv('tring_data.csv', delimiter=',', header=0)
X_train = training_data.rop('event',1)  
X_train.head()

#  X_train=X_train.values
y_train = training_data[['event']]  # y_train=y_train.values
X_test = test_data.drop('event',1) # X_test=X_test.values
y_test = test_data[['event']] # y_test=y_test.values

#-------- Classificataion Models: DT, RF -------------#
# ML algorithm
models = []
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Random Forest', RandomForestClassifier()))


# running algorithms
for name, model in models:
    train_model = model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acc = accuracy_score(prediction, y_test)
    roc = roc_auc_score(prediction, y_test)
    msg = "===========%s===========\n accuracy : %f \n roc_auc: %f" % (name, acc, roc)
    print(msg)

## 의사결정나무 시각화
## GraphVIz 설치 필요: https://graphviz.gitlab.io/download/

export_graphviz(models[0][1], out_file="tree.dot")

os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin"
export_graphviz(models[0][1], out_file="./tree.dot", class_names=["True", "False"])


with open("./tree.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='tree', directory='./', cleanup=True)





















