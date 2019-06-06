#================================================================
# Theano 
#----------------------------------------------------------------
# 是一个用于定义数学函数（对向量和矩阵进行操作）的Python库，并计算这些函数的梯度
# Theano是Keras等许多深度学习软件包的基础层
# Theano 允许用户定义编码损失函数的数学表达式，只要表达式被定义，Theano允许用户计算这些表达式的梯度
#----------------------------------------------------------------


#================================================================
# 用标量的函数
#----------------------------------------------------------------
# 1. 标量在用于数学表达式前被定义
# 2. 每个标量都有一个独特的名字
# 3. 一旦定义，标量可以用+ - * / 等操作
# 4. Theano中的函数构造允许关联输入和输出
import numpy
import theano
import theano.tensor as T 
from theano import function

# a = T.dscalar('a')
# b = T.dscalar('b')
# c = T.dscalar('c')
# d = T.dscalar('d')
# e = T.dscalar('e')
# f = ((a-b+c)*d)/e
# g = function([a,b,c,d,e],f)
# print('Expected: (1-2+3)*4/5 = ', (1-2+3)*4/5)
# print('Theano: (1-2+3)*4/5 = ', g(1,2,3,4,5))

#----------------------------------------------------------------


#================================================================
# 用向量的函数
#----------------------------------------------------------------
# a = T.dmatrix('a')
# b = T.dmatrix('b')
# c = T.dmatrix('c')
# d = T.dmatrix('d')
# e = (a+b-c)*d
# f = function([a,b,c,d], e)

# a_data = numpy.array([[1,1], [1,1]])
# b_data = numpy.array([[2,2], [2,2]])
# c_data = numpy.array([[5,5], [5,5]])
# d_data = numpy.array([[3,3], [3,3]])

# print('Expected:', (a_data+b_data-c_data)*d_data)
# print('Theano:', f(a_data,b_data,c_data,d_data))
#----------------------------------------------------------------


#================================================================
# 用向量和标量的函数
#----------------------------------------------------------------
# a = T.dmatrix('a')
# b = T.dmatrix('b')
# c = T.dmatrix('c')
# d = T.dmatrix('d')

# p = T.dscalar('p')
# q = T.dscalar('q')
# r = T.dscalar('r')
# s = T.dscalar('s')
# u = T.dscalar('u')

# e = (((a*p)+(b-q)-(c+r))*d/s)*u

# f = function([a,b,c,d,p,q,r,s,u], e)

# a_data = numpy.array([[1,1], [1,1]])
# b_data = numpy.array([[2,2], [2,2]])
# c_data = numpy.array([[5,5], [5,5]])
# d_data = numpy.array([[3,3], [3,3]])

# print('Expected:', (((a_data*1.0)+(b_data-2)-(c_data+3))*d_data/4)*5)
# print('Theano:', f(a_data,b_data,c_data,d_data,1,2,3,4,5))

#----------------------------------------------------------------


#================================================================
# 激活函数
#----------------------------------------------------------------
# Theano中的nnet包定义了许多常用的激活函数
#----------------------------------------------------------------
# # sigmoid
# a = T.dmatrix('a')
# f_a = T.nnet.sigmoid(a)
# f_sigmoid = function([a],[f_a])
# print('sigmoid:', f_sigmoid([[-1,0,1]]))

# # tanh
# b = T.dmatrix('b')
# f_b = T.tanh(b)
# f_tanh = function([b],[f_b])
# print('tanh:', f_tanh([[-1,0,1]]))

# # fast sigmoid
# c = T.dmatrix('c')
# f_c = T.nnet.ultra_fast_sigmoid(c)
# f_fast_sigmoid = function([c],[f_c])
# print('fast sigmoid:', f_fast_sigmoid([[-1,0,1]]))

# # softplus
# d = T.dmatrix('d')
# f_d = T.nnet.softplus(d)
# f_softplus = function([d],[f_d])
# print('soft plus:',f_softplus([[-1,0,1]]))

# # relu
# e = T.dmatrix('e')
# f_e = T.nnet.relu(e)
# f_relu = function([e],[f_e])
# print('relu:', f_relu([[-1,0,1]]))

# # softmax
# f = T.dmatrix('f')
# f_f = T.nnet.softmax(f)
# f_softmax = function([f],[f_f])
# print('soft max:', f_softmax([[-1,-0,1]]))

#================================================================
# 共享变量
#----------------------------------------------------------------
from theano import shared

# x = T.dmatrix('x')
# y = shared(numpy.array([[4,5,6]]))
# z = x + y
# f = function(inputs=[x], outputs=[z])
# print('Original Shared Value:', y.get_value())
# print('Original Function Evaluation:', f([[1,2,3]]))

# y.set_value(numpy.array([[5,6,7]]))

# print('Original Shared Value:', y.get_value())
# print('Original Function Evaluation:', f([[1,2,3]]))

#----------------------------------------------------------------


#================================================================
# 梯度
#----------------------------------------------------------------
# x = T.dmatrix('x')
# y = shared(numpy.array([[4,5,6]]))
# z = T.sum(((x*x)+y)*x)
# f = function(inputs=[x], outputs=[z])

# g = T.grad(z, [x])

# g_f = function([x], g) # 梯度函数

# print('Original:', f([[1,1,1]]))
# print('Original Gradient:', g_f([[1,1,1]]))
#----------------------------------------------------------------


#================================================================
# 损失函数
#----------------------------------------------------------------
# # 二元交叉熵
# a1 = T.dmatrix('a1')
# a2 = T.dmatrix('a2')
# f_a = T.nnet.binary_crossentropy(a1, a2).mean()
# f_sigmoid = function([a1, a2], [f_a])
# print('Binary Cross Entropy:', f_sigmoid([[0.01,0.01,0.01]],[[0.99,0.99,0.01]]))

# # 分类交叉熵
# b1 = T.dmatrix('b1')
# b2 = T.dmatrix('b2')
# f_b = T.nnet.categorical_crossentropy(b1, b2)
# f_sigmoid = function([b1, b2], [f_b])
# print('Categorical Cross Entropy:', f_sigmoid([[0.01,0.01,0.01]],[[0.99,0.99,0.01]]))

# 平方差
# def squared_error(x,y):
# 	return (x-y)**2

# c1 = T.dmatrix('c1')
# c2 = T.dmatrix('c2')
# f_c = squared_error(c1, c2)
# f_squared_error = function([c1,c2],[f_c])
# print('Square Error:', f_squared_error([[0.01,0.01,0.01]],[[0.99,0.99,0.01]]))
#----------------------------------------------------------------


#================================================================
# 正则项
#----------------------------------------------------------------
# # L1 正则项
# def l1(x):
# 	return T.sum(abs(x))

# # L2 正则项
# def l2(x):
# 	return T.sum(x**2)

# a = T.dmatrix('a')
# f_a = l1(a)
# f_l1 = function([a], f_a)
# print('L1 Regularization:', f_l1([[0,-1,3]]))

# b = T.dmatrix('b')
# f_b = l2(b)
# f_l2 = function([b], f_b)
# print('L2 Regularization:', f_l2([[0,-1,3]]))

#----------------------------------------------------------------


#================================================================
# RandomStreams 构造函数，它允许使用随机变量来定义函数 使用种子进行初始化
#----------------------------------------------------------------
# from theano.tensor.shared_randomstreams import RandomStreams
# import random

# seed = random.randint(1,10)
# random = RandomStreams(seed=42)

# a = random.normal((1,3))
# b = T.dmatrix('b')

# f1 = a*b
# g1 = function([b], f1)

# print('Invocation 1:', g1(numpy.ones((1,3))))
# print('Invocation 2:', g1(numpy.ones((1,3))))
# print('Invocation 3:', g1(numpy.ones((1,3))))

#----------------------------------------------------------------


#================================================================
# 逻辑回归
#----------------------------------------------------------------
import sklearn.metrics

# def l2(x):
# 	return T.sum(x**2)

# examples = 1000
# features = 100

# D = (numpy.random.randn(examples, features), numpy.random.randint(size=examples, low=0, high=2)) # 只有D[0] 和 D[1] D[1]由0,1组成
# training_steps = 1000

# x = T.dmatrix('x')
# y = T.dvector('y')
# w = theano.shared(numpy.random.randn(features), name='w')
# b = theano.shared(0.0, name='b')

# p = 1 / (1+T.exp(-T.dot(x,w)-b))
# error = T.nnet.binary_crossentropy(p, y)
# loss = error.mean() + 0.01*l2(w)
# prediction = p > 0.5
# gw, gb = T.grad(loss, [w,b])

# train = theano.function(inputs=[x,y], outputs=[p,error], updates=((w,w-0.1*gw),(b,b-0.1*gb)))
# predict = theano.function(inputs=[x], outputs=prediction)

# print('Accuracy before Training:', sklearn.metrics.accuracy_score(D[1], predict(D[0])))

# for i in range(training_steps):
# 	prediction, error = train(D[0], D[1])

# print('Accuracy after Training:', sklearn.metrics.accuracy_score(D[1], predict(D[0])))

#----------------------------------------------------------------


#================================================================
# 线性回归
#----------------------------------------------------------------
# def l2(x):
# 	return T.sum(x**2)

# def square_error(x,y):
# 	return (x-y)**2

# examples = 1000
# features = 100

# D = (numpy.random.randn(examples, features), numpy.random.randn(examples)) 
# training_steps = 1000

# x = T.dmatrix('x')
# y = T.dvector('y')
# w = theano.shared(numpy.random.randn(features), name='w')
# b = theano.shared(0.0, name='b')

# p = T.dot(x,w) + b
# error = square_error(p, y)
# loss = error.mean() + 0.01*l2(w)
# gw, gb = T.grad(loss, [w,b])

# train = theano.function(inputs=[x,y], outputs=[p,error], updates=((w,w-0.1*gw),(b,b-0.1*gb)))
# predict = theano.function(inputs=[x], outputs=p)

# # RMSE 均方根误差
# print('RMSE before Training:', sklearn.metrics.mean_squared_error(D[1], predict(D[0])))

# for i in range(training_steps):
# 	prediction, error = train(D[0], D[1])

# print('RMSE after Training:', sklearn.metrics.mean_squared_error(D[1], predict(D[0])))
#----------------------------------------------------------------


#================================================================
# 神经网络
#----------------------------------------------------------------
# def l2(x):
# 	return T.sum(x**2)

# examples = 1000
# features = 100
# hidden = 10

# D = (numpy.random.randn(examples, features), numpy.random.randint(size=examples, low=0, high=2))
# training_steps = 1000

# x = T.dmatrix('x')
# y = T.dvector('y')

# w1 = theano.shared(numpy.random.randn(features, hidden), name='w1')
# b1 = theano.shared(numpy.zeros(hidden), name='b1')

# w2 = theano.shared(numpy.random.randn(hidden), name='w2')
# b2 = theano.shared(0.0, name='b2')

# p1 = T.tanh(T.dot(x, w1) + b1)
# p2 = T.tanh(T.dot(p1, w2) + b2)

# prediction = p2 > 0.5
# error = T.nnet.binary_crossentropy(p2, y)

# loss = error.mean() + 0.01*(l2(w1) + l2(w2))
# gw1, gb1, gw2, gb2 = T.grad(loss, [w1,b1,w2,b2])

# train = theano.function(inputs=[x,y], outputs=[p2,error], updates=((w1,w1-0.1*gw1),(b1,b1-0.1*gb1),(w2,w2-0.1*gw2),(b2,b2-0.1*gb2)))
# predict = theano.function(inputs=[x], outputs=prediction)

# print('Accuracy before Training:', sklearn.metrics.accuracy_score(D[1], predict(D[0])))

# for i in range(training_steps):
# 	prediction, error = train(D[0], D[1])

# print('Accuracy after Training:', sklearn.metrics.accuracy_score(D[1], predict(D[0])))

#----------------------------------------------------------------


#================================================================
# switch/ if-else
#----------------------------------------------------------------
# 可以使用if-else swtich结构定义表达式和函数 并且可以像其他表达式一样生成梯度 
# IfElse takes a boolean condition and two variables as inputs.
# Switch takes a tensor as condition and two variables as inputs. 
# Switch is an elementwise operation and is thus more general thanifelse.

# from theano.ifelse import ifelse

# def hinge_a(x,y):
# 	return T.max([0*x, 1-x*y])

# def hinge_b(x,y):
# 	return ifelse(T.lt(1-x*y,0), 0*x, 1-x*y) # output =  x(a<b)	y(a>=b)

# def hinge_c(x,y):
# 	return T.switch(T.lt(1-x*y,0), 0*x, 1-x*y)

# x = T.dscalar('x')
# y = T.dscalar('y')

# z1 = hinge_a(x,y)
# z2 = hinge_b(x,y)
# z3 = hinge_c(x,y)

# f1 = theano.function([x,y], z1)
# f2 = theano.function([x,y], z2)
# f3 = theano.function([x,y], z3)

# print('f(-2,1)=', f1(-2,1), f2(-2,1), f3(-2,1))
# print('f(1,1)=', f1(1,1), f2(1,1), f3(1,1))
#----------------------------------------------------------------


#================================================================
# scan
#----------------------------------------------------------------
# 使用scan来实现乘方操作 并和使用标准库的乘方函数得到的结果相同

import theano.printing

k = T.iscalar('k')
a = T.dscalar('a')

result, updates = theano.scan(fn=lambda prior_result, a:prior_result*a, outputs_info=a, non_sequences=a, n_steps=k-1)
final_result = result[-1]
a_pow_k = theano.function(inputs=[a,k], outputs=final_result, updates=updates)
print(a_pow_k(2,5))

#----------------------------------------------------------------

