import numpy as np


# Gradient Descent for Multiple Variables 多参数梯度下降的代价函数
def computeCost(X, y, theta):
	inner = np.power(((X * theta.T) - y), 2)
	return np.sum(inner) / (2 * len(X))

# Normal Equation 正规方程
def normalEqn(X, y):
	theta = np.linalg.inv(X.T@X)@X.T@y # X.T@X == X.T.dot(X)
	return theta

