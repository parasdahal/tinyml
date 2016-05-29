import numpy as np

class linear_regression():
	"""
	Compute linear regression problems on single or multiple features
	"""
	def __init__(self,table):
		self.num_training= np.shape(table)[0]
		self.features=np.delete(table,-1,1)
		self.features=np.insert(self.features,0,np.ones(self.num_training),axis=1)	#Add a new column in front of features
		self.num_features = np.shape(self.features)[1]
		self.values = table[:,self.num_features-1]
		self.theta = np.ones(self.num_features)
		
	def compute_cost(self):
		"""Computes cost based on the current values of the parameters
		
		"""
		hypothesis = np.dot(self.features,self.theta)
		loss = hypothesis - self.values
		return np.sum(loss ** 2)/ (2*self.num_training)
	
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
			hypothesis = np.dot(self.features,self.theta)
			loss = hypothesis - self.values
			cost = self.compute_cost()
			print "Iteration: %d Cost: %f" % (i,cost)
			gradient = np.dot(self.features.T,loss) / self.num_training
			self.theta = self.theta - alpha*gradient
		return self.theta
		
	def predict(self,data):
		"""Computes the value based on current weights
		
		Parameters
		----------
		data : ndarray 
			Test data to score using the current weights
		Returns
		-------
		val : float
			Prediction result
		"""
		data = np.insert(data,0,np.ones(1))
		return np.dot(data,self.theta)
	
	def plot_data(self):
		if self.num_features==2:
			from matplotlib import pyplot as plt
			x = self.features[:,-1]
			y = self.values
			plt.scatter(x,y)
			plt.show()
	
	def plot_fit(self):
		if self.num_features==2:
			from matplotlib import pyplot as plt
			xi=np.arange(self.features[:,-1].min()-1,self.features[:,-1].max()+1)
			line = self.theta[0]+(self.theta[1]*xi)
			plt.plot(xi,line,'r-',self.features[:,-1],self.values,'o')
			plt.show()
