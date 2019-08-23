import numpy as np
import random
import scipy.special
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# learning rate
learning_rate = 0.2


# 首先，读取.CSV文件成矩阵的形式。
my_matrix = np.loadtxt(open("Hu.csv"),delimiter=",",skiprows=0)
# 对于矩阵而言，将矩阵倒数第一列之前的数值给了X（输入数据），将矩阵大最后一列的数值给了y（标签）
X, y = my_matrix[:,:-1],my_matrix[:,-1]
# 利用train_test_split方法，将X,y随机划分问，训练集（X_train），训练集标签（X_test），测试卷（y_train）测试集标签（y_test），安训练集：测试集=7:3的概率划分，到此步骤，可以直接对数据进行处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 此步骤，是为了将训练集与数据集的数据分别保存为CSV文件
# np.column_stack将两个矩阵进行组合连接
train= np.column_stack((X_train,y_train))
# numpy.savetxt 将txt文件保存为。csv结尾的文件
# np.savetxt('train_usual.csv',train, delimiter = ',')
test = np.column_stack((X_test, y_test))
# np.savetxt('test_usual.csv', test, delimiter = ',')

# scaler = StandardScaler() # 标准化转换
# scaler.fit(X_train)  # 训练标准化对象
# X_t = scaler.transform(X_train)   # 转换数据集

# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9, 3), random_state=1)
clf.fit(X_train, y_train)

print('预测结果：', clf.predict([[0.5, 0, 0]]))  # 预测某个输入对象

cengindex = 0
for wi in clf.coefs_:
    cengindex += 1  # 表示底第几层神经网络。
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵：\n',wi)

