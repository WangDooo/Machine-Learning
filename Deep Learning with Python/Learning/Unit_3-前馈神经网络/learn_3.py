#================================================================
# 用AutoGrad创建神经网络 
# AutoGrad是一个自动求导库，他可以自动计算出用Numpy编写的函数的梯度
#----------------------------------------------------------------
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sklearn.metrics
import pylab

# 生成数据集
examples = 1000
features = 100
D = (npr.randn(examples, features), npr.randn(examples))

# 指定网络
layer1_units = 10
layer2_units = 1
w1 = npr.rand(features, layer1_units)
b1 = npr.rand(layer1_units)
w2 = npr.rand(layer1_units, layer2_units)
b2 = 0.0
theta = (w1, b1, w2, b2)

# 定义损失函数
def squared_loss(y, y_hat):
	return np.dot((y-y_hat), (y-y_hat))

# 输出层
def binary_cross_entropy(y, y_hat):
	return np.sum()
#----------------------------------------------------------------

