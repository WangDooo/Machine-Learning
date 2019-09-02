#================================================================
# Autograd
#----------------------------------------------------------------
# 是一个自动求导工具的例子，它重载已建立的Numpy库，而Theano提供自己的操作符，为其执行相应的求导操作
#----------------------------------------------------------------


#================================================================
# Autograd实现反向模式自动求导，并可以计算任意Python和Numpy代码的导数
#----------------------------------------------------------------
# import autograd.numpy as numpy
# from autograd import grad

# # 定义函数
# def f(x1, x2):
# 	return numpy.sqrt(x1*x1+x2*x2)

# # 计算对于第一个输入变量x1的导数
# g_x1_f = grad(f,0)

# # 计算对于第二个输入变量x2的导数
# g_x2_f = grad(f,1)

# # 在x1=1, x2=2处评测并打印数值
# print(f(1,2))

# # 在x1=1, x2=2处评测并打印x1的导数
# print(g_x1_f(1.0,2.0))


# # 在x1=1, x2=2处评测并打印x2的导数
# print(g_x2_f(1.0,2.0))


#----------------------------------------------------------------


#================================================================
# 用Autograd进行逻辑回归
#----------------------------------------------------------------
import pylab
import sklearn.datasets
import autograd.numpy as np 
from autograd import grad

# 生成数据
train_X, train_y = sklearn.datasets.make_moons(500, noise=0.1) # 生成半环形图

# 为逻辑回归定义激活、预测和损失函数
def activation(x):
	return 0.5*(np.tanh(x)+1)

def predict(weights, inputs):
	return activation(np.dot(inputs, weights))

def loss(weights):
	preds = predict(weights, train_X)
	label_probabilities = preds * train_y + (1-preds)*(1-train_y)
	return -np.sum(np.log(label_probabilities))

# 计算损失函数的梯度
gradient_loss = grad(loss)

# 设定初始值
weights = np.array([1.0, 1.0])

# 最速下降
loss_values = []
learning_rate = 0.001
for i in range(100):
	loss_values.append(loss(weights))
	step = gradient_loss(weights)
	weights -= step*learning_rate

# 画出决策边界
x_min, x_max = train_X[:, 0].min()-0.5, train_X[:,0].max()+0.5
y_min, y_max = train_X[:, 1].min()-0.5, train_X[:,1].max()+0.5
x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = predict(weights, np.c_[x_mesh.ravel(), y_mesh.ravel()])
Z = Z.reshape(x_mesh.shape)
cs = pylab.contourf(x_mesh, y_mesh, Z, cmap=pylab.cm.Spectral)
pylab.scatter(train_X[:,0], train_X[:,1], c=train_y, cmap=pylab.cm.Spectral)
pylab.colorbar(cs)

# 画进一步的损失
pylab.figure()
pylab.plot(loss_values)
pylab.xlabel('Steps')
pylab.ylabel('Loss')
pylab.show()

#----------------------------------------------------------------
