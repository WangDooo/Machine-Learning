import numpy as np 

def normalEqn(X, y):
	theta = np.linalg.inv(X.T@X)@X.T@y # X.T@X == X.T.dot(X)
	return theta