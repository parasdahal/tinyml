import numpy as np

class LogisticRegression:
    """
    Classification using logistic regression
    """

    def __init__(self, table,reg=False,lamda=0,degree = 1):
        """Initializes Class for Logistic Regression
        
        Parameters
        ----------
        table : ndarray(n-rows,m-features + 1)
            Numerical training data, last column as training labels
        reg : Boolean
            Set True to enable regularization, false by default
        degree: int
            Degree of polynomial to fit to the data. Default is 1.
            
        """
        # Regularization Parameters
        self.reg=reg
        self.lamda=lamda
        # training data
        self.degree=degree
        self.table = table
        # num of rows in the training data
        self.num_training = np.shape(table)[0] 
        # map the features according to the degree of the fit
        self.X = self.map_features()
        self.num_features = np.shape(self.X)[1]
        # extract last column from training data table
        self.y = table[:,-1]
        # craete an array of parameters initialzing them to 1
        self.theta = np.zeros(self.num_features)
        
    def map_features(self):
        """
        Generates polynomial features based on the degree
        """
        X = self.table[:,0]
        Y = self.table[:,1]
        # First column is ones for calculation of intercept
        features = np.ones(self.num_training)
        
        for i in range(1,self.degree+1):
                for j in range(0,i+1):
                    col = np.power(X,i-j)*np.power(Y,j)
                    features = np.column_stack((features,col))
        return features
        
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
        #new ndarray to prevent intercept from theta array to be changed
        theta=np.delete(self.theta,0)
        #regularization term
        reg = (self.lamda/2*self.num_training)*np.sum(np.power(theta,2)) 
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
        old_cost=0
        for i in range(0, num_iters):
            hypothesis = LogisticRegression.sigmoid(np.dot(self.X, self.theta))
            loss = hypothesis - self.y
            cost = self.compute_cost()
            #if the cost is equal for two iterations, break from the loop
            if cost==old_cost:
                break
            old_cost = cost
            #print "Iteration: %d Cost: %f" % (i, cost)
            gradient = np.dot(self.X.T, loss) / self.num_training
            #regularization term
            reg = (self.lamda/self.num_training)*self.theta
            if self.reg:
                self.theta = self.theta - alpha * (gradient+reg)
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
        plt.scatter(self.X[:, 1], self.X[:, 2], s=40, c=self.y, cmap=plt.cm.Spectral)
        plt.show()

    def plot_fit(self):
        """Plot the training data in X array along with decision boundary
        """
        from matplotlib import pyplot as plt
        x1 = np.linspace(self.table.min(), self.table.max(), 100)
        #reverse self.theta as it requires coeffs from highest degree to constant term
        x2 = np.polyval(np.poly1d(self.theta[::-1]),x1)
        plt.plot(x1, x2, color='r', label='decision boundary');
        plt.scatter(self.X[:, 1], self.X[:, 2], s=40, c=self.y, cmap=plt.cm.Spectral)
        plt.legend()
        plt.show()
