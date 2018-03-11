import numpy as np 
import matplotlib.pyplot as plt

# 数据初始化
# a0 = np.random.normal()
# a1 = np.random.normal()
a0 = 5
a1 = 5
# 学习步长
rate = 0.001

def h(x):	# h(x)=a0+a1x
    return a0+a1*x

rate = 0.001
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

for i in range(50000): # 设定迭代次数
    sum_a0=0
    sum_a1=0
    for x, y in zip(x_train, y_train):
        sum_a0 = sum_a0 + rate*(y-h(x))*1
        sum_a1 = sum_a1 + rate*(y-h(x))*x
    a0 = a0 + sum_a0
    a1 = a1 + sum_a1
    #plt.plot([h(xi) for xi in x_train])

print(a0, a1)
result=[h(xi) for xi in x_train]
print(result)

plt.show()

# -------网上找的----------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# #y=2 * (x1) + (x2) + 3 
# rate = 0.001
# x_train = np.array([[1, 2],[2, 1],[2, 3],[3, 5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7]])
# y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
# x_test  = np.array([[1, 4],[2, 2],[2, 5],[5, 3],[1, 5],[4, 1]])

# a = np.random.normal()
# b = np.random.normal()
# c = np.random.normal()

# def h(x):
#     return a*x[0]+b*x[1]+c

# for i in range(10000):
#     sum_a=0
#     sum_b=0
#     sum_c=0
#     for x, y in zip(x_train, y_train):
#         sum_a = sum_a + rate*(y-h(x))*x[0]
#         sum_b = sum_b + rate*(y-h(x))*x[1]
#         sum_c = sum_c + rate*(y-h(x))
#     a = a + sum_a
#     b = b + sum_b
#     c = c + sum_c
#     plt.plot([h(xi) for xi in x_test])

# print(a)
# print(b)
# print(c)

# result=[h(xi) for xi in x_train]
# print(result)

# result=[h(xi) for xi in x_test]
# print(result)

# plt.show()