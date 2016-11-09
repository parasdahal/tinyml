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

			z = activation.dot(weight)
			activation = afunc(z)
			zs.append(z)
			activations.append(activation)

		return (zs,activations)

	def train(self,X,y,alpha,reg):
		
		weights = list() # list of weights per each layer
		biases = list() #list of biases for each layer

		for i in range(1,len(self.shape)):
			# size of weight matrix for each layer => (num of units in this layer * num of units in next layer)
			weights.append(np.random.random_sample((self.shape[i-1],self.shape[i])))
			biases.append(np.zeros((1,self.shape[i])))
		
		temp_weight = weights
		temp_bias = biases
		# iterate over all rows in the dataset
		for x,y in zip(X,y):

			zs,activations = self.predict(x)
			#last layer
			delta = activations[-1] - y
			temp_weight[-1] = np.dot(delta,activations[-1].T)
			temp_bias[-1] = delta.sum()
			# peroform backward pass for each layer except last
			for l in range(1,len(self.shape)):
				delta = np.dot(delta,self.weights[-l]) * (activations[-l]*(1-activations[-l]))
				temp_weight[-l] = delta.dot(activations[-l].T)
				temp_bias[-l] = delta.sum()

			# gradient descent parameter update for weights
			for weight,dWeight in zip(weights,temp_weight):
				# apply regularization to the weight
				dWeight += reg * weight
				weight += -alpha*dWeight

			# gradient descent parameter update for biases
			for bias,d in zip(weights,temp_bias):
				bias += -alpha*d
		
		self.weights = weights
		self.biases = biases
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
	
	# def plot_decision_boundary(self,X,y):
	# 	# Set min and max values and give it some padding
	# 	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	# 	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	# 	h = 0.01
	# 	# Generate a grid of points with distance h between them
	# 	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# 	# Predict the function value for the whole gid
	# 	Z = self.predict(np.c_[xx.ravel(), yy.ravel()])[1][-1]
	# 	print(Z.shape)
	# 	Z = Z.reshape(xx.shape)
	# 	# Plot the contour and training examples
	# 	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	# 	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	# 	plt.show()


if __name__ == "__main__":
	from sklearn import datasets
	X,y = datasets.make_moons(200,noise=0.2)
	print(np.shape(X))
	print((X[1],y[1]))
	NN = FullyConnectedNN([2,2,1],X,y)
	NN.plot_data()
	print(NN.train(X,y,alpha=0.3,reg=0.1))
	NN.predict(X[1])
	# NN.plot_decision_boundary(X,y)