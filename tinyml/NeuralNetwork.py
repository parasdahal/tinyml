import numpy as np 

class NeuralNetwork:
	"""
		General Neural Network implementation
	"""

	def __init__(self,shape,activation="sigmoid"):
		"""Initializes a neural network model

		Parameters
		----------

		shape: ndarray()
			A list of integers specifying number of nodes on each layer starting from input to output layer
		activation: string
			The activation function to be used. Currently only sigmoid is implemented
		"""
		self.shape = shape	# list storing the structure of the network
		self.activation = activation	# activation function to be used
		self.weights = list()	# list of weights per each layer
		for i in range(1,len(shape)):
			# creation of parameter matrices for each layer mapping with random values
			# size of weight matrix for each layer => (num of units in next layer * num of units in that layer+1)
			self.weights.append(np.random.random_sample((shape[i],shape[i-1]+1)))	



NN = NeuralNetwork(np.array([2,3,3,2]))