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

# 梯度检验 检验反向传播的量级是否正确 【gradApprox 与 Dvec(反向传播得到的对theta的偏导)】的比较
# 因为在反向传播中会存在一些bug 
# 此检验程序运算量大耗时 检验过后需turn off
def CompareGradApproxWithDvec(theta, EPSILON):
	for i in range(n):
		thetaPlus = theta
		thetaPlus[i] = thetaPlus[i] + EPSILON
		thetaMinus = theta
		thetaMinus[i] = thetaMinus[i] + EPSILON
		gradApprox[i] = (J(thetaPlus) - J(thetaMinus)) / (2*EPSILON) # J()代价函数