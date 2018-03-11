# h(x) = a0 + a1x
# 五个样本 1,1	2,2.2	3,3.1	4,4.8	5,4.7

import numpy as np 
import matplotlib.pyplot as plt

# 数据初始化
# a0 = np.random.normal()
# a1 = np.random.normal()
a0 = 5
a1 = 5
# 学习步长
rate = 0.001

def h(x):
    return a0+a1*x

rate = 0.001
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

for i in range(50000):
    sum_a0=0
    sum_a1=0
    for x, y in zip(x_train, y_train):
        sum_a0 = sum_a0 + rate*(y-h(x))*1
        sum_a1 = sum_a1 + rate*(y-h(x))*x
    a0 = a0 + sum_a0
    a1 = a1 + sum_a1
    #plt.plot([h(xi) for xi in x_train])

print(a0)
print(a1)

result=[h(xi) for xi in x_train]
print(result)

plt.show()
