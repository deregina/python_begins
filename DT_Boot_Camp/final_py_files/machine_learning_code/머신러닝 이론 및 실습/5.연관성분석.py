# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:50:37 2019

@author: pns_com2
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from apyori import apriori 

store_data = pd.read_csv('store_data.csv', header=None)  
store_data.head()  

records = []  
for i in range(0, 7501):  
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2,      				                  min_lift=3, min_length=2)  


association_results = list(association_rules) 

print(len(association_results))
print(association_results[0]) 

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    
     
#second index of the inner list
    print("Support: " + str(item[1]))

#third index of the list located at 0th
#of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    
    
    
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

store_data = []
file = open("store_data.csv", "r")
for line in file:
    temp = line.strip().split(",")
    store_data.append(temp)
    
te = TransactionEncoder()
te_ary = te.fit(store_data).transform(store_data)

df = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5) 
pd.DataFrame(association_rules(frequent_itemsets, metric="lift", min_threshold= 2)).head()



