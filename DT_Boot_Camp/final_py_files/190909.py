# -*- coding: utf-8 -*-

a1 = 3
a2 = 3.5
a3 = "abc"
type(a1)
a4 = True
True + False

# 리스트 구조
b1 = [4, 2, 5]
b1[0]
b1[1] = 7
print(b1)

b2 = (4, 7, 2)

b6 = [[1, 2, 3], [1, 2], [3]]
b6[0][1]
b6[1:3]

b7 = [2, 5, 4, 7, 1, 9, 3]
b7[:]
b7[0:3]
b7[-1]
b7[:-2]
b7[-9]
b7[-7]
b7[::2]
b7[3::2]
b7[0:5:2]
c7 = list.copy(b7)
c7
c7.append(3)
c7
d7 = b7.copy()
d7

b7 > 5
a1 > 5
b7 * b7
3 * b7
b7 + b1
c1 = b7[:4]  # 복사기능
c2 = b7      # 참조기능
b7
b7[2] = 8
c2
c1
c3 = b7.copy()
b7[2] = 6
c3
b7
c2
c1
b7.append()
b7.pop()
b7
b7.remove(3)
b7.remove(9)
b7
b7.append(9)
b7.reverse()
b7
b8 = "초밥이랑 회를 먹고 싶어!"
b8[0]
b8.count("어")
b8.count("랑", 3, 8)
b8.count("랑", 4, 8)
b8.find("랑")
b8.index("감")
b2 + b2
b8.find("dk")
b8.sort()
b1.sort()
sorted(b1)
sorted(b1, reverse = True)

# {"key" : 상수 / List / Tuple / Dictionary, "Key" : 값} or dict()
d1 = {"Age" : [3, 5, 9], "Height" : [60, 70, 80]}
type(d1)
d1[0]

d1["Height"]
d1["Height"][1]
d1.pop("Age")
d1
d1.popitem()
d1
d1 = {"Age" : [3, 5, 9], "Height" : [60, 70, 80]}
d1.keys()
d1.values()
d1.items()
d1["weight"] = [30, 40, 20]
d1


# 폴더명.함수명
import ft1
ft1.add2(5, 7)
ft1.add1(2, 3)

# 함수명
from ft1 import add1, add2
# from ft1 import *
add1(3, 4)
add1(3, 4, 2)

# numpy 구조, array 구조
import numpy as np
f1 = np.array(b7)
b7 > 5
b7[0] > 5
f1 > 5
f1
sum(b7)
sum(f1)
type(b7)
type(f1)
print(b7)
print(b7)
print(f1)
f1.shape
f2 = np.array([[2, 5, 7], 
               [5, 6, 2]])
f2
f2 * f2
np.matmul(f2, f2.T)
f2.shape
f2[0]
f2[0][1]
f1[f1>5]
f2
np.sum(f2, 0)
np.sum(f2, 1)
np.argmax(f2, 0)
np.argmax(f2,1)

# np.where(조건, 참인 경우 실행할 문장, 거짓인 경우 실행할 문장)
np.where(f1>5, 1, 0)
f3 = np.where(f2>5, 1, 0)
f3
f4 = range(1, 5) # 5 미포함, 0 ~ 4
f4
f5 = np.arange(5)
f5
f6 = np.zeros((2, 3))
f6
f7 = np.ones(8)
f7
f8 = np.ones((3, 4))
f8

f2 = f2.reshape(3, 2)
f2
