import numpy as np


class LinearRegression:
    """
    Compute linear regression problems on single or multiple features
    """

    def __init__(self, table):
        self.num_training = np.shape(table)[0]
        # remove the last column from training data to extract features data
        self.X = np.delete(table, -1, 1)
        # add a column of ones in front of the training data
        self.X = np.insert(self.X, 0, np.ones(self.num_training), axis=1)
        self.num_features = np.shape(self.X)[1]
        # extract the values of the training set from the provided data
        self.y = table[:, self.num_features - 1]
        # create parameters and initialize to 1
        self.theta = np.ones(self.num_features)

    def compute_cost(self):
        """Computes cost based on the current values of the parameters using square cost function
        """
        hypothesis = np.dot(self.X, self.theta)
        loss = hypothesis - self.y
        return np.sum(loss ** 2) / (2 * self.num_training)

    def gradient_descent(self, num_iters=1000, alpha=0.01):
        """Runs the gradient decent algorithm
        
        Parameters
        ----------
        num_iters : int
            The maximum number of iterations allowed to run before the algorithm terminates
        alpha : float
            The learning rate for the algorithm
            
        """
        for i in range(0, num_iters):
            hypothesis = np.dot(self.X, self.theta)
            loss = hypothesis - self.y
            cost = self.compute_cost()
            print "Iteration: %d Cost: %f" % (i, cost)
            gradient = np.dot(self.X.T, loss) / self.num_training
            self.theta = self.theta - alpha * gradient
        return self.theta

    def predict(self, data):
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
        # add a column of ones to the prediction data to make it compatible with parameter array
        data = np.insert(data, 0, np.ones(1))
        return np.dot(data, self.theta)

    def plot_data(self, feature=1):
        """Plots the training data with feature number in parameter as x axis (deafault = 1)
        
        Parameters
        ----------
        feature : int
                  number of the parameter to be represented in x axis 
        """
        from matplotlib import pyplot as plt
        x = self.X[:, feature]
        y = self.y
        plt.scatter(x, y)
        plt.show()

    def plot_fit(self, feature=1):
        """Plots the training data with regression line. The feature number as x axis is specified in parameter
        
        Parameters
        ----------
        feature : int
                  number of the parameter to be represented in x axis 
        """
        from matplotlib import pyplot as plt
        xi = np.arange(self.X[:, feature].min() - 1, self.X[:, feature].max() + 1)
        line = self.theta[0] + (self.theta[1] * xi)
        plt.plot(xi, line, 'r-', self.X[:, feature], self.y, 'o')
        plt.show()
