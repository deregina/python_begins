# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:43:08 2019

@author: user
"""

# 자료 종류, 자료구조
# 리스트, 튜플, 딕셔너리
# [], (), {"키": 값, "키": 값}
# list(), tuple(), dict("키": 값, "키": 값)
# 상수 단위 처리 x0 = 3
# 접근 : 
# - 숫자접근 [index 방식] 변수명[번호(0 ~)], x1[0], x1[0][1]
# - 키 방식 : 변수명["키"] x2["Age"], x2["Age"][2]

x1 = [[[[[1, 2], 3], 3], 4, [5, 8]], [5, 7, [1, 2]], 9]
len(x1)
x1[1]           # [5, 7, [1, 2]]
x1[1][2]        # [1, 2]
x1[1][2][0]     # 1
x1.append(5)
x1
x2 = (3, 5, 2, (5, 2, 5, (7, 4)))
len(x2)         # 4
type(x2)        # tuple, append 사용 불가
x2[3]           # (5, 2, 5, (7, 4))
x2[3][3]        # (7, 4)
x2[3][3][0]     # 7
type(x2[3][3])  # tuple
x3 = {"age" : [3, 5], "name" : ["Kim", "Lee", "Ha"]}
type(x3)
x3["age"]
x3["age"][0]
x3.keys()
x3.values()
x3["height"] = [170, 163, 183]
x3
# x3 = {"age" : [3, 5], 
#       "name" : ["Kim", "Lee", "Ha"], 
#       "height" : [170, 163, 183]}

x0 = {"weight" : [15, 20, 68]}
x3.update(x0)
x3
x3.update(age = [10, 9, 8])
# x3.update(이미 존재하는 키 = 바꾸고 싶은 값)

x3

x1 > 5          # Error
x1[0] > 5       # Error
x1[0][0] > 5    # 상수 단위만 처리 가능

x4 = [5, 3, 2, 7, 6, 9, 2]
x4 > 5

for i in x4 :
    print(i)
# 요소 프린트
    
for i in range (len(x4)) :
    print(x4[i])
# 요소 프린트
    
for i in range (len(x4)) :
    print(i)
# 0~6 프린트 
    
for i in range (len(x4)) :
    print(x4[i] > 5)
print(i)
# 들여쓰기가 들어간 곳까지 한 문장
# 들여쓰기 들어간 곳 작업이 끝나야 넘어갈 수 있음
# False False False True True True False 6

for i in range (len(x4)) :      # [0, 1, 2, 3, 4, 5, 6]
    print(x4[i] > 5)
    print(i)
# 들여쓰기가 있으므로 모두 한 문장
# False 0 False 1 False 2 True 3 True 4 True 5 False 6
    

range(1, 5)                     # 1, 2, 3, 4  출력은 안됨
for i in range(1, 5):
    print(i)
    
np.arange(1, 5)                 # array([1, 2, 3, 4]) 출력
np.arange(5)
np.arange(0, 10, 3)
    
print(np.arange(len(x4)))       # [0 1 2 3 4 5 6]
np.arange(len(x4))              # array([0, 1, 2, 3, 4, 5, 6])

for i in np.arange(len(x4)):
    print(i)

import numpy as np
for i in np.arange(len(x4)) :   # [0, 1, 2, 3, 4, 5, 6]
    print(x4[i] > 5)


#  중첩 for loop
for i in x1 :
    for j in i :
        for k in j :
            print(k)
            
#  x1 때문에 실행은 끝까지 안된다
for i in np.arange(len(x1)) :
    for j in np.arange(len(x1[i])) :
        for k in np.arange(len(x1[i][j])) :
            print(x1[i][j][k])
            
# in range로 하는 거랑 차이가 없지만 
# 출력해서 확인해 보려면 np.arange로 해보는 게 낫다    
x5 = [[1, 2, 3], [4, 5, 6]]
for i in np.arange(len(x5)):
    for j in np.arange(len(x5[i])) :
        print(x5[i][j])
        
# 들여쓰기 끝나는 부분 주의
for i in x5 :
    for j in i :
        print(j, end = "\t")    
        
for i in x5 :
    for j in i :
        print(j, end = "\t")
        print()
        
for i in x5 :
    for j in i :
        print(j, end = "\t")
    print()
    
# if 조건문
# 조건이 참인 경우 명령 실행
# continue : 밑에 실행 안하고 skip
for i in x5 :
    for j in i :
        if j == 5 : continue
        print(j, end = "\t")  
        
# break : loop 종료
for i in x5 :
    for j in i :
        if j == 5 : break
        print(j, end = "\t")  
  
# *를 변수 앞에 쓰면 갯수를 안 맞춰도
# 나머지 요소들이 다 그 변수에 들어감        
a1, a2 = 3, 4
a3 = 3, 4
a4, a5 = 3, 4, 7 
a7, *a8, a9 = 3, 4, 7, 9, 10
print(a4)
print(a5)
print(a7)
print(a8)
print(a9)


# Unpacking   분리해서 받기
family = {"dad" : "homer", "mom" : "marge", "daughter" : "Lisa"}
for key, value in family.items() :
    print(key, value)
    
for i, j in x3.items() :
    print(j)
    
