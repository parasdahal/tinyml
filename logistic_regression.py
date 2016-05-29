import numpy as np

class logistic_regression():
	
	def __init__(self,table):
		"""Initializes Class for Logistic Regression
		
		Parameters
		----------
		X : ndarray(n-rows,m-features)
			Numerical training data.
		y: ndarray(n-rows,)
			Interger training labels.
			
		"""
		self.table=table
		self.num_training = np.shape(table)[0]
		self.features = np.delete(table,-1,1)
		self.features = np.insert(self.features,0,np.ones(self.num_training),axis=1)
		print self.features
		self.num_features = np.shape(self.features)[1]
		self.values = table[:,self.num_features-1]
		self.theta = np.ones(self.num_features)
	
	@staticmethod
	def sigmoid(val):
		"""Computes sigmoid function of input value
		
		Parameters
		----------
		val : float
			  Value of which sigmoid is to calculate
		
		"""
		return 1/(1+np.exp(-val))

	def compute_cost(self):
		"""Computes cost based on the current values of the parameters
		
		"""
		hypothesis = logistic_regression.sigmoid(np.dot(self.features,self.theta))
		cost = -(np.sum(self.values*np.log(hypothesis)+(1-self.values)*(np.log(1-hypothesis))))/self.num_training
		return cost
		
	def gradient_descent(self,num_iters=1000,alpha=0.01):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		num_iters : int
			The maximum number of iterations allowed to run before the algorithm terminates
		alpha : float
			The learning rate for the algorithm
			
		"""
		for i in range(0,num_iters):
			hypothesis = logistic_regression.sigmoid(np.dot(self.features,self.theta))
			loss = hypothesis - self.values
			cost = self.compute_cost()
			print "Iteration: %d Cost: %f" % (i,cost)
			gradient = np.dot(self.features.T,loss) / self.num_training
			self.theta = self.theta - alpha*gradient
		return self.theta
	
	def predict(self,data):
		"""Computes the logistic probability of being a positive example
		
		Parameters
		----------
		data : ndarray (n-rows,n-features)
			Test data to score using the current weights
		Returns
		-------
		0 or 1: int
			0 if probablity is less than 0.5, else 1
		"""
		data = np.insert(data,0,np.ones(1))
		hypothesis = logistic_regression.sigmoid(np.dot(data,self.theta))
		return hypothesis
			
	def plot_data(self):
			if self.num_features == 3:
				from matplotlib import pyplot as plt
				plt.scatter(self.features[:,-2],self.features[:,-1], s = 40, c = self.values, cmap = plt.cm.Spectral )
			plt.show()
			
	def plot_fit(self):
		if self.num_features==3:
			from matplotlib import pyplot as plt
			x1=np.arange(self.features[:,-2].min()-1,self.features[:,-2].max()+1)
			x2=np.arange(self.features[:,-1].min()-1,self.features[:,-1].max()+1)
			line = self.theta[0]+self.theta[1]*x1+self.theta[2]*x2
			plt.plot(line,'r-')
			plt.scatter(self.features[:,-2],self.features[:,-1], s = 40, c = self.values, cmap = plt.cm.Spectral )
			plt.show()
