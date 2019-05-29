#================================================================
# 泛化 根据所拥有的数据量来调整模型容量 欠拟合 过拟合
#----------------------------------------------------------------
# 生成示例数据集
import pylab
import numpy

x = numpy.linspace(-1,1,100)
signal = 2 + x + 2*x*x
noise = numpy.random.normal(0,0.1,100) # 均值0 标准差0.1 
y = signal + noise
pylab.plot(signal,'b')
pylab.plot(y,'g')
pylab.plot(noise,'r')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.title("Init Data")
pylab.legend(['Without Noise', "With Noise", "Noise"], loc=2)
x_train = x[0:80]
y_train = y[0:80]

# 模型度为
def degree_run(degree):
	pylab.figure()
	X_train = numpy.column_stack([numpy.power(x_train,i) for i in range(0,degree)])
	# 最小二乘法
	model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(),X_train)),X_train.transpose()),y_train)
	pylab.plot(x,y,'g')
	pylab.xlabel("x")
	pylab.ylabel("y")
	title = "Degree = "+ str(degree)
	pylab.title(title)
	predicted = numpy.dot(model, [numpy.power(x,i) for i in range(0,degree)])
	pylab.plot(x, predicted, 'r')
	pylab.legend(["Actual", "Predicted"], loc=2)
	train_rmsel = numpy.sqrt(numpy.sum(numpy.dot(y[0:80]-predicted[0:80], y_train-predicted[0:80])))
	test_rmsel = numpy.sqrt(numpy.sum(numpy.dot(y[80:]-predicted[80:], y[80:]-predicted[80:])))
	print("Train RMSE (Degree = ",degree,")", train_rmsel)
	print("Test RMSE (Degree = ",degree,")", test_rmsel)

degree_run(2)
degree_run(3)
degree_run(9)
pylab.show()

#----------------------------------------------------------------


#================================================================
# 正规化 核心思想是惩罚模型的复杂度
#----------------------------------------------------------------

#----------------------------------------------------------------
