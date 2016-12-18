import numpy as np
import random

class NN:

	def __init__(self,sizes):

		self.sizes = sizes
		self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
		self.biases = [np.random.randn(x,1) for x in sizes[1:]]

	def forwardprop(self,a):

		for w,b in zip(self.weights,self.biases):
			z = np.dot(w,a) + b
			a = self.sigmoid(z)
		return a

	def train(self,training_data,epoch,batch_size):
		print('Starting Training')
		print('Batch Size:',batch_size)
		for current_epoch in range(epoch):
			random.shuffle(training_data)
			print('Training Epoch:',current_epoch+1)
			batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
			for batch in batches:
				self.update_batch(batch)

	def update_batch(self,batch):

		ALPHA = 0.1
		weight = [np.zeros(w.shape) for w in self.weights]
		bias = [np.zeros(b.shape) for b in self.biases]

		for X,y in batch:
			delta_w , delta_b = self.backprop(X,y)
			weight = [wb+w for wb,w in zip(delta_w,weight)]
			bias = [db+b for db,b in zip(delta_b,bias)]
		
		self.biases = [b - (ALPHA)/len(batch)*nb for b,nb in zip(self.biases,bias)]
		self.weights = [w - (ALPHA/len(batch))*nw for w,nw in zip(self.weights,weight)]

	def backprop(self,X,y):

		w = [np.zeros(w.shape) for w in self.weights]
		b = [np.zeros(b.shape) for b in self.biases]

		activation = X
		activations = [X]
		zs = []
		# forward pass
		for we,be in zip(self.weights,self.biases):
			z=np.dot(we,activation)+be
			activation = self.sigmoid(z)
			zs.append(z)
			activations.append(activation)

		# output layer error
		delta = self.cost_derrivative(activations[-1],y)*self.sigmoid_prime(zs[-1])
		w[-1] = np.dot(delta,activations[-2].transpose())
		b[-1] = delta

		# propagate throuh rest of the layers
		for l in range(2,len(self.sizes)):
			z = zs[-l]
			delta = np.dot(self.weights[-l+1].transpose(),delta)*self.sigmoid_prime(z)
			b[-l] = delta
			w[-l] = np.dot( delta, activations[-l-1].transpose())

		return w,b

	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def cost_derrivative(self,a,y):
		return a-y


from sklearn import datasets

digits = datasets.load_digits()
training_data = [(x,y) for x,y in zip(digits['data'],digits['target'])]

net = NN([64,64,10])
net.train(training_data,10,100)