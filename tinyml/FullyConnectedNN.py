import numpy as np 

class FullyConnectedNN:
	"""
		General Fully connected Neural Network implementation
	"""

	def __init__(self,shape,X,y,activationfunc="sigmoid"):
		"""Initializes a neural network model

		Parameters
		----------

		shape: list
			A list of integers specifying number of nodes on each layer starting from input to output layer
		activation: string
			The activation function to be used. Currently only sigmoid is implemented
		"""
		self.X = X
		self.y = y
		self.shape = shape	# list storing the structure of the network
		self.activationfunc = activationfunc	# activation function to be used
		
		self.weights = list() # list of weights per each layer
		self.biases = list() #list of biases for each layer

		for i in range(1,len(shape)):
			# size of weight matrix for each layer => (num of units in this layer * num of units in next layer)
			self.weights.append(np.random.random_sample((shape[i-1],shape[i])))
			self.biases.append(np.zeros((1,shape[i])))
		

	def predict(self,X):
		""" Performs forward propagation
		
		Parameters
		----------

		X: ndarray
			A numpy array with values for input layer
		"""
		# store the computations
		zs = []
		# store the activations
		activations = [X]
		# initial activation is the input
		activation = X
		# get activation function
		afunc = self.activation_func()

		for weight,bias in zip(self.weights,self.biases):

			z = activation.dot(weight) + bias
			activation = afunc(z)
			zs.append(z)
			activations.append(activation)

		return (zs,activations)

	def train(self,alpha):
		
		weights = list() # list of weights per each layer
		biases = list() #list of biases for each layer

		for i in range(1,len(shape)):
			# size of weight matrix for each layer => (num of units in this layer * num of units in next layer)
			self.weights.append(np.random.random_sample((shape[i-1],shape[i])))
			self.biases.append(np.zeros((1,shape[i])))
		
		temp_weight = weights
		temp_bias = biases
		# iterate over all rows in the dataset
		for x,y in zip(self.X,self.y):

			zs,activations = self.predict(x)
			#last layer
			delta = activations[-1] - y
			temp_weight[-1] = np.dot(delta,activations[-1])
			temp_bias[-1] = delta.sum()
			# peroform backward pass for each layer except last
			for l in range(1,len(self.shape)):
				delta = np.dot(delta,self.weights[-l]) * (activations[-l]*(1-activations[-l]))
				temp_weight[-l] = delta.activations[-l]
				temp_bias[-l] = delta.sum()

			for weight,d in zip(weights,temp_weight):
				weight += -alpha*d
			
			for bias,d in z(weights,temp_bias):
				bias += -alpha*d
		
		return (weights,biases)


	def activation_func(self):
		""" Returns the activation function given in the constructor parameter

		Returns
		-------
		function: function
			The activation function
		"""
		def sigmoid(val):
			return float(1)/(1+np.exp(-val))
		
		if(self.activationfunc == "sigmoid"):
			return sigmoid

	def plot_data(self):

		from matplotlib import pyplot as plt
		plt.scatter(self.X[:,0],self.X[:,1],s=40, c=y, cmap=plt.cm.Spectral)
		plt.show()

if __name__ == "__main__":
	from sklearn import datasets
	X,y = datasets.make_moons(200,noise=0.2)
	print(np.shape(X))
	NN = FullyConnectedNN([2,2,2,2],X,y)
	NN.plot_data()
	print(NN.predict(X[1]))
	# NN.back_propagation()