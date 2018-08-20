import numpy as np


# Gradient Descent for Multiple Variables 多参数梯度下降的代价函数
def computeCost(X, y, theta):
	inner = np.power(((X * theta.T) - y), 2)
	return np.sum(inner) / (2 * len(X))

# Normal Equation 正规方程
def normalEqn(X, y):
	theta = np.linalg.inv(X.T@X)@X.T@y # X.T@X == X.T.dot(X)
	return theta

# logistic function 逻辑函数
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# 逻辑函数的代价函数
def cost(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
	return np.sum(first - second) / (len(X))

# 正则化的逻辑回归模型 的代价函数
def costReg(theta, X, y, learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
	second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
	reg = (learningRate / (2*len(X))*np.sum(np.power(theta[:,1:theta.shape[1]],2)))
	return np.sum(first - second) / (len(X)) + reg

