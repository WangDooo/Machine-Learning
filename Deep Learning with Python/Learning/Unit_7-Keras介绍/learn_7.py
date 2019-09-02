#================================================================
# 单层神经网络
#----------------------------------------------------------------
# 使用Sequential构造函数定义模型 允许用户添加配置模型
# Dense层基本上是一个全连接层
# 定义第一层时需要制定输入和输出维度
# 模型定义后需要编译，提供计算的损失模型、优化算法、其他指标
# 需要选择适当的损失函数，通常是随机梯度下降的变体
# 编译完成后，可以通过提供数据和评估方法来建立模型

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical

# model = Sequential()
# model.add(Dense(1, input_dim=500))
# model.add(Activation(activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# data = np.random.random((1000,500))
# labels = np.random.randint(2, size=(1000,1)) # 1000个1维的 0或1
# score = model.evaluate(data, labels, verbose=0) 
# # verbose：日志显示
# # verbose = 0 为不在标准输出流输出日志信息
# # verbose = 1 为输出进度条记录
# # verbose = 2 为每个epoch输出一行记录
# print('Before Training:', list(zip(model.metrics_names, score)))

# model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
# # epochs指的就是训练过程接中数据将被“轮”多少次”
# # Keras中参数更新是按批进行的，就是小批梯度下降算法，把数据分为若干组，称为batch，
# # 按批更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，
# # 一批数据中包含的样本数量称为batch_size。

# score = model.evaluate(data, labels, verbose=0) 
# print('After Training:', list(zip(model.metrics_names, score)))
# plot_model(model, to_file='s1.png', show_shapes=True)
#----------------------------------------------------------------


#================================================================
# 两层神经网络
#----------------------------------------------------------------
# 使用Dense定义第二层，不需要指定维度，以为与上一层输出的维度相同

# model = Sequential()
# model.add(Dense(32, input_dim=500)) # 第一层 输入500 输出32
# model.add(Activation(activation='sigmoid'))
# model.add(Dense(1)) # 第二层 输入随上层32 输出 1
# model.add(Activation(activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# data = np.random.random((1000,500))
# labels = np.random.randint(2, size=(1000,1))

# score = model.evaluate(data, labels, verbose=0)
# print('Before Training:', list(zip(model.metrics_names, score)))

# model.fit(data, labels, epochs=10, batch_size=32, verbose=0)

# score = model.evaluate(data, labels, verbose=0) 
# print('After Training:', list(zip(model.metrics_names, score)))
# plot_model(model, to_file='s2.png', show_shapes=True)
#----------------------------------------------------------------


#================================================================
# 用于多元分类的两层神经网络
#----------------------------------------------------------------
# 使用Dense定义第二层，将输出维度定义为10 这是与数据集中的类别数完全相等

# model = Sequential()
# model.add(Dense(32, input_dim=500))
# model.add(Activation(activation='relu'))
# model.add(Dense(10))
# model.add(Activation(activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# data = np.random.random((1000,500))
# labels = to_categorical(np.random.randint(10, size=(1000,1)))

# score = model.evaluate(data, labels, verbose=0)
# print('Before Training:', list(zip(model.metrics_names, score)))

# model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
# score = model.evaluate(data, labels, verbose=0)
# print('After Training:', list(zip(model.metrics_names, score)))

# plot_model(model, to_file='s3.png', show_shapes=True)

#----------------------------------------------------------------


#================================================================
# 两层神经网络的回归
#----------------------------------------------------------------

# model = Sequential()
# model.add(Dense(32, input_dim=500))
# model.add(Activation(activation='sigmoid'))
# model.add(Dense(1))
# model.add(Activation(activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_squared_error'])

# data = np.random.random((1000,500))
# labels = np.random.randint(2, size=(1000,1))

# score = model.evaluate(data, labels, verbose=0)
# print('Before Training:', list(zip(model.metrics_names, score)))

# model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
# score = model.evaluate(data, labels, verbose=0)
# print('After Training:', list(zip(model.metrics_names, score)))

# plot_model(model, to_file='s4.png', show_shapes=True)
#----------------------------------------------------------------


#================================================================
# Keras快速迭代 比较激活算法 优化器
#----------------------------------------------------------------
# def train_given_optimiser(optimiser):
# 	model = Sequential()
# 	model.add(Dense(1, input_dim=500))
# 	model.add(Activation(activation='sigmoid'))
# 	model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])

# 	data = np.random.random((1000,500))
# 	labels = np.random.randint(2, size=(1000,1))

# 	score = model.evaluate(data, labels, verbose=0)
# 	print('Optimiser:', optimiser)
# 	print('Before Training:', list(zip(model.metrics_names, score)))

# 	model.fit(data, labels, epochs=10, batch_size=32, verbose=0)

# 	score = model.evaluate(data, labels, verbose=0)
# 	print('After Training:', list(zip(model.metrics_names, score)))

# train_given_optimiser('sgd')
# train_given_optimiser('rmsprop')
# train_given_optimiser('adagrad')
# train_given_optimiser('adadelta')
# train_given_optimiser('adam')
# train_given_optimiser('adamax')
# train_given_optimiser('nadam')
#----------------------------------------------------------------


#================================================================
# 激活函数
#----------------------------------------------------------------
# def train_given_activation(activation):
# 	model = Sequential()
# 	model.add(Dense(1, input_dim=500))
# 	model.add(Activation(activation=activation))
# 	model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# 	data = np.random.random((1000,500))
# 	labels = np.random.randint(2, size=(1000,1))

# 	score = model.evaluate(data, labels, verbose=0)
# 	print('Activation:', activation)
# 	print('Before Training:', list(zip(model.metrics_names, score)))

# 	model.fit(data, labels, epochs=10, batch_size=32, verbose=0)

# 	score = model.evaluate(data, labels, verbose=0)
# 	print('After Training:', list(zip(model.metrics_names, score)))

# train_given_activation('relu')
# train_given_activation('tanh')
# train_given_activation('sigmoid')
# train_given_activation('hard_sigmoid')
# train_given_activation('linear')

#----------------------------------------------------------------


#================================================================
# 使用Keras构建卷积神经网络 CNN
#----------------------------------------------------------------
from keras.layers import Dropout, Flatten, Embedding
from keras.layers import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils

# # 图像大小
# img_rows, img_cols = 28, 28

# # 滤波器
# nb_filters = 32
# # 池化
# pool_size = (2,2)

# # 核
# kernel_size = (3,3)

# # 准备数据集
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# nb_classes = 10
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# # CNN
# model = Sequential()
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# # 编译
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# # 训练
# batch_size = 128
# epochs = 1
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))

# # 评测
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test Metrics:', list(zip(model.metrics_names, score)))
# plot_model(model, to_file='s7.png', show_shapes=True)

#----------------------------------------------------------------


#================================================================
# 使用Keras构建LSTM
#----------------------------------------------------------------
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80
batch_size = 32
epochs = 1

# 准备数据
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# LSTM
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 编译
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

# 评测
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test Metrics:', list(zip(model.metrics_names, score)))
plot_model(model, to_file='s8.png', show_shapes=True)
#----------------------------------------------------------------

