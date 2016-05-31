import numpy as np

class LogisticRegression:
    """
    Classification using logistic regression
    """

    def __init__(self, table,reg=False,lamda=0):
        """Initializes Class for Logistic Regression
        
        Parameters
        ----------
        table : ndarray(n-rows,m-features + 1)
            Numerical training data, last column as training labels
        reg : Boolean
            Set True to enable regularization, false by default
            
        """
        #Regularization Parameters
        self.reg=reg
        self.lamda=lamda
        #training data
        self.table = table
        self.num_training = np.shape(table)[0]  #num of rows in the training data
        self.X = np.delete(table, -1, 1)    #remove last column from traing data to get feature data
        self.X = np.insert(self.X, 0, np.ones(self.num_training), axis=1)   #add a column of ones to feature data
        self.num_features = np.shape(self.X)[1]
        self.y = table[:, self.num_features - 1]    #extract last column from training data table
        self.theta = np.zeros(self.num_features)    #craete an array of parameters initialzing them to 1

    @staticmethod
    def sigmoid(val):
        """Computes sigmoid function of input value

        Parameters
        ----------
        val : float
              Value of which sigmoid is to calculate
        Returns
        -------
        val : float
            Sigmoid value of the parameter
        
        """
        return float(1) / (1 + np.exp(-val))

    def compute_cost(self):
        """Computes cost based on the current values of the parameters
        
        Returns
        -------
        cost : float
            Cost of the selection of current set of parameters
        
        """
        hypothesis = LogisticRegression.sigmoid(np.dot(self.X, self.theta))
        #regularization term
        reg = (self.lamda/2*self.num_training)*np.sum(np.power(self.theta,2)) 
        cost = -(np.sum(self.y * np.log(hypothesis) + (1 - self.y) * (np.log(1 - hypothesis)))) / self.num_training
        #if regularization is true, add regularization term and return cost
        if self.reg:
            return cost + reg
        return cost

    def gradient_descent(self, num_iters=1000, alpha=0.01):
        """Runs the gradient descent algorithm
        
        Parameters
        ----------
        num_iters : int
            The maximum number of iterations allowed to run before the algorithm terminates
        alpha : float
            The learning rate for the algorithm
        
        Returns
        -------
        self.theta: ndarray(self.features)
            Array of parameters after running the algorithm
        """
        for i in range(0, num_iters):
            hypothesis = LogisticRegression.sigmoid(np.dot(self.X, self.theta))
            loss = hypothesis - self.y
            cost = self.compute_cost()
            print "Iteration: %d Cost: %f" % (i, cost)
            gradient = np.dot(self.X.T, loss) / self.num_training
            #regularization term
            reg = (1 - (self.lamda*alpha)/self.num_training)
            if self.reg:
                self.theta = self.theta*reg - alpha * gradient
            else:
                self.theta = self.theta - alpha * gradient
        return self.theta

    def predict(self, data, prob=False):
        """Computes the logistic probability of being a positive example
        
        Parameters
        ----------
        data : ndarray (n-rows,n-features)
            Test data to score using the current weights
        prob : Boolean
            If set to true, probability will be returned, else binary classification
        Returns
        -------
        0 or 1: int
            0 if probablity is less than 0.5, else 1
        """
        data = np.column_stack((np.ones(data.shape[0]), data))

        hypothesis = LogisticRegression.sigmoid(np.dot(data, self.theta))
        if not prob:
            return np.where(hypothesis >= .5, 1, 0)
        return hypothesis

    def accuracy(self):
        """Calculates percentage of correct predictions by the model on training data
        
        Returns
        -------
        accuracy : float
            Percentage of correct predictions on the features of training data
        """
        #delete extra ones column that was added
        x = np.delete(self.X, 0, 1)
        predicted = self.predict(x)
        match = float(np.sum(self.y == predicted))
        return (match / self.num_training) * 100

    def plot_data(self):
        """Plot the training data in X array
        """
        from matplotlib import pyplot as plt
        plt.scatter(self.X[:, -2], self.X[:, -1], s=40, c=self.y, cmap=plt.cm.Spectral)
        plt.show()

    def plot_fit(self):
        """Plot the training data in X array along with decision boundary
        """
        from matplotlib import pyplot as plt
        x1 = np.linspace(self.X.min()-1, self.X.max()+1, 100)
        x2 = -(self.theta[1] * x1 + self.theta[0]) / self.theta[2]
        plt.plot(x1, x2, color='r', label='decision boundary');
        plt.scatter(self.X[:, -2], self.X[:, -1], s=40, c=self.y, cmap=plt.cm.Spectral)
        plt.legend()
        plt.show()
