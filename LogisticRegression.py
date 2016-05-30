import numpy as np

class LogisticRegression:
    """
    Classification using logistic regression
    """

    def __init__(self, table):
        """Initializes Class for Logistic Regression
        
        Parameters
        ----------
        X : ndarray(n-rows,m-features)
            Numerical training data.
        y: ndarray(n-rows,)
            Interger training labels.
            
        """
        self.table = table
        self.num_training = np.shape(table)[0]
        self.X = np.delete(table, -1, 1)
        self.X = np.insert(self.X, 0, np.ones(self.num_training), axis=1)
        print self.X
        self.num_features = np.shape(self.X)[1]
        self.y = table[:, self.num_features - 1]
        self.theta = np.zeros(self.num_features)

    @staticmethod
    def sigmoid(val):
        """Computes sigmoid function of input value

        Parameters
        ----------
        val : float
              Value of which sigmoid is to calculate
        
        """
        return float(1) / (1 + np.exp(-val))

    def compute_cost(self):
        """Computes cost based on the current values of the parameters
        
        """
        hypothesis = LogisticRegression.sigmoid(np.dot(self.X, self.theta))
        cost = -(np.sum(self.y * np.log(hypothesis) + (1 - self.y) * (np.log(1 - hypothesis)))) / self.num_training
        return cost

    def gradient_descent(self, num_iters=1000, alpha=0.01):
        """Runs the gradient descent algorithm
        
        Parameters
        ----------
        num_iters : int
            The maximum number of iterations allowed to run before the algorithm terminates
        alpha : float
            The learning rate for the algorithm
            
        """
        for i in range(0, num_iters):
            hypothesis = LogisticRegression.sigmoid(np.dot(self.X, self.theta))
            loss = hypothesis - self.y
            cost = self.compute_cost()
            print "Iteration: %d Cost: %f" % (i, cost)
            gradient = np.dot(self.X.T, loss) / self.num_training
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
        """Returns percentage of correct predictions by the model on training data
        """
        x = np.delete(self.X, 0, 1)
        predicted = self.predict(x)
        match = float(np.sum(self.y == predicted))
        return (match / self.num_training) * 100

    def plot_data(self):
        """Plot the training data in X array
        """
        if self.num_features == 3:
            from matplotlib import pyplot as plt
            plt.scatter(self.X[:, -2], self.X[:, -1], s=40, c=self.y, cmap=plt.cm.Spectral)
        plt.show()

    def plot_fit(self):
        """Plot the training data in X array along with decision boundary
        """
        if self.num_features == 3:
            from matplotlib import pyplot as plt
            x1 = np.linspace(self.X.min()-1, self.X.max()+1, 100)
            x2 = -(self.theta[1] * x1 + self.theta[0]) / self.theta[2]
            plt.plot(x1, x2, color='r', label='decision boundary');
            plt.scatter(self.X[:, -2], self.X[:, -1], s=40, c=self.y, cmap=plt.cm.Spectral)
            plt.legend()
            plt.show()
