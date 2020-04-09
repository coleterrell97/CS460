import numpy as np
from csv import reader

trainFileName = "./data/mnist_train_0_1.csv"
testFileName  = "./data/mnist_test_0_1.csv"
learningRate = 0.5

def sigmoid(x):
	return 1/(1+np.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def openCSV(fileName):
	with open(fileName, 'r') as csvFile:
		csvData = []
		csvFileReader = reader(csvFile)
		for row in csvFileReader:
			csvData.append(row)
		csvData = np.array(csvData, dtype = "float")
			#normalization step
		csvData[0][1:] = csvData[0][1:] - csvData[0][1:].mean(axis=0)
		csvData[0][1:] = csvData[0][1:] / csvData[0][1:].max(axis=0)
	return csvData

class NeuralNetwork:
	def __init__(self, trainData, testData, *dimensions):
		self.layers = []
		self.weights = []
		self.trainData = trainData
		self.testData = testData
		#initialize values at each layer
		for dimension in dimensions:
			self.layers.append(np.ones(dimension))
		self.layers = np.array(self.layers)
		#initialize weights between each layer
		for layerNum in range(1, self.layers.shape[0]):
			self.weights.append(np.random.rand(self.layers[layerNum].shape[0], self.layers[layerNum - 1].shape[0])-0.5)
	def __forward__(self, iMatrix, iToJWeights):
		return sigmoid_v(np.matmul(iToJWeights, iMatrix))


	def __backprop__(self, example):
		for layer in range(self.layers.shape[0] - 1, 0, -1):
			if layer == self.layers.shape[0] - 1:
				delta = (example[0] - self.layers[layer]) * (self.layers[layer]) * (1 - self.layers[layer])
			else:
				delta = (self.layers[layer]) * (1 - self.layers[layer]) * np.matmul(np.transpose(self.weights[layer]), delta)
			self.weights[layer - 1] = self.weights[layer - 1] + learningRate * np.matmul(delta, np.transpose(self.layers[layer - 1]))

	def train(self):
		for example in self.trainData:
			#orient input matrix correctly
			for i in range(1, 785):
				self.layers[0][i][0] = example[i]
			#pass the input into the neural network going forward
			for layer in range(1, self.layers.shape[0]):
				self.layers[layer] = self.__forward__(self.layers[layer - 1], self.weights[layer-1])
			self.__backprop__(example)

	def test(self):
		numCorrect = 0
		print("Incorrect predictions:")
		for example in self.testData:
			for i in range(1, 785):
				self.layers[0][i][0] = example[i]
			for layer in range(1, self.layers.shape[0]):
				self.layers[layer] = self.__forward__(self.layers[layer - 1], self.weights[layer-1])
			if self.layers[-1] > 0.8:
				prediction = 1
			elif self.layers[-1] < 0.2:
				prediction = 0
			else:
				prediction = -1
			if prediction == example[0]:
				numCorrect = numCorrect + 1
			else:
				print(self.layers[-1], example[0])
		print("Accuracy: ", numCorrect / self.testData.shape[0] * 100)



#define a neural net of the desired dimensions
NN = NeuralNetwork(openCSV(trainFileName), openCSV(testFileName), (785,1), (200,1), (1,1))
NN.train()
NN.test()
