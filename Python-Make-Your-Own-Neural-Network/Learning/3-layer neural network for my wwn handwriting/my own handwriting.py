import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image

# nerual network class definition
class neuralNetwork:

	# initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		# learning rate
		self.lr = learningrate
		# link weight matrices, wih and who
		# weights inside the array are w_i_y, where link is from node i to node j in the next layer
		self.wih = (np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
		self.who = (np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		pass 

	# train the neural network
	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		
		# error is the (target - actual)
		output_errors = targets - final_outputs

		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)
		
		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
		pass

	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# calculate signals into final output layer
		final_inputs = np.dot(self.who,hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
		
		return final_outputs

	def save_weights(self):
		np.savetxt("result_wih.txt", self.wih)
		np.savetxt("result_who.txt", self.who)

	def load_weights(self):
		self.wih = np.loadtxt("result_wih.txt")
		self.who = np.loadtxt("result_who.txt")


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# learning rate
learning_rate = 0.2
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the mnist training data CSV file into a list
# training_data_file = open("mnist_dataset/mnist_train_100.csv",'r')
training_data_file = open("F:/Books/《Python神经网络编程》中文版PDF+英文版PDF+源代码/makeyourownneuralnetwork源代码r/makeyourownneuralnetwork-master/mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
	# Test
	# image_array = np.asfarray(all_values[1:]).reshape((28,28))
	# plt.imshow(image_array, cmap='Greys', interpolation='None')
# go through all records in the training data set


# load the mnist test data CSV file into a list
test_data_file = open("F:/Books/《Python神经网络编程》中文版PDF+英文版PDF+源代码/makeyourownneuralnetwork源代码r/makeyourownneuralnetwork-master/mnist_dataset/mnist_test.csv",'r')
# test_data_file = open("mnist_dataset/mnist_test_10.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# epochs is the number of times the training data set is used for training
# epochs = 5

# for e in range(epochs):
# 	for record in training_data_list:
# 		all_values = record.split(',')
# 		# scale and shift the inputs
# 		inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
# 		# create the target output values (all 0.01, except the desired label which is 0.99)
# 		targets = np.zeros(output_nodes) + 0.01
# 		targets[int(all_values[0])] = 0.99
# 		n.train(inputs,targets)
# 		pass

# n.save_weights()


# n.load_weights()
# # test the neural network
# scorecard = []
# # go through all the records in the test data set
# for record in test_data_list:
# 	all_values = record.split(',')
# 	correct_label = int(all_values[0])
# 	# print("correct_label:",correct_label)
# 	# scale and shift the inputs
# 	inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
# 	# query the network
# 	outputs = n.query(inputs)
# 	# the index of the highest value corresponds to the label
# 	label = np.argmax(outputs)
# 	# print("answer:",label)
# 	# append correct or incorrect to list
# 	if(label == correct_label):
# 		scorecard.append(1)
# 	else:
# 		scorecard.append(0)
# 		pass
# # print(scorecard)
# scorecard_array = np.asarray(scorecard)
# print("performance = ",scorecard_array.sum()/scorecard_array.size)

file = 'test_5.png'
new_file = 'new'+ file
img = Image.open(file)
try:
	new_img = img.resize((28,28),Image.BILINEAR)   
	new_img.save(new_file)
except Exception as e:
	print(e)

img_array = scipy.misc.imread(new_file, flatten=True)

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0*0.99)+0.01

n.load_weights()
inputs = np.asfarray(img_data)
outputs = n.query(inputs)
print(outputs)
label = np.argmax(outputs)
print("answer = ", label)