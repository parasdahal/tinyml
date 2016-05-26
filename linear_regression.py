import numpy as np

class linear_regression():
	"""
	Compute linear regression problems on single or multiple features
	"""
	def __init__(self,table):
		self.num_training= np.shape(table)[0]
		self.features=np.delete(table,-1,1)
		self.features=np.insert(self.features,0,np.ones(self.num_training),axis=1)	#Add a new column in front of features
		x , self.num_features = np.shape(self.features)
		self.values = table[:,self.num_features-1]
		self.theta = np.ones(self.num_features)
		
	def compute_cost(self):
		hypothesis = np.dot(self.features,self.theta)
		loss = hypothesis - self.values
		cost = np.sum(loss ** 2)/ (2*self.num_training)
		return cost
	
	def gradient_descent(self,num_iters=100,alpha=0.1):
		for i in range(0,num_iters):
			hypothesis = np.dot(self.features,self.theta)
			loss = hypothesis - self.values
			cost = self.compute_cost()
			print "Iteration: %d Cost: %f" % (i,cost)
			gradient = np.dot(self.features.T,loss) / self.num_training
			self.theta = self.theta - alpha*gradient
		return self.theta
		
	def predict(self,data):
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
			xi=np.arange(0,10)
			line = self.theta[0]+(self.theta[1]*xi)
			plt.plot(xi,line,'r-',self.features[:,-1],self.values,'o')
			plt.show()
