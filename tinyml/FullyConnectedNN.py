import numpy as np
import random
import math
import datetime

class CrossEntropyCost:

    @staticmethod
    def fn(a,y):
        return np.sum( np.nan_to_num( -y * np.log(a) - (1-y) * np.log(1-a) ) )
    
    @staticmethod
    def delta(a,y):
        return (a-y)

class FullyConnectedNN:

    def __init__(self, sizes, cost=CrossEntropyCost):
        # TODO: Get datasets here and parallelize them into RDD
        logger.info("Starting up network")
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialize_weights()
        self.cost = cost
    
    def initialize_weights(self):
        """Initializing weights as Gaussian random variables with mean
        0 and standard deviation 1/sqrt(n) where n is the number
        of weights connecting to the same neuron.

        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feed_forward(self,a):
        
        for b,w in zip(self.biases,self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a
    
    def backprop(self,x,y):
        
        # biases and weights calculated by backprop
        b = [np.zeros(bias.shape) for bias in self.biases]
        w = [np.zeros(weight.shape) for weight in self.weights]
        
        # forward pass
        activation = x
        activations = [x]
        zs = []
        for bias,weight in zip(self.biases,self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # output error
        delta = (self.cost).delta(activations[-1],y)
        b[-1] = delta
        w[-1] = np.dot(delta,activations[-2].transpose())

        # backpropagate
        for l in xrange(2,self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            # store the derrivative terms in the bias and weight list
            b[-l] = delta
            w[-l] = np.dot(delta,activations[-l-1].transpose())
        
        return (b,w)
    
    def gd_mini_batch(self,mini_batch,alpha,lmbda,n):
        """Update the weights and biases of the netwrok by applying
        gradient descent on each mini batch. Mini batch is a list
        of tuple (x,y)

        """
        biases = [np.zeros(b.shape) for b in self.biases]
        weights = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            # get derrivative terms using backprop
            # TODO: delta_b,delta_w = mini_batch.map(lambda x,y: self.backprop(x,y))
            # store these deltas and accumulate weights and biases
            delta_b, delta_w = self.backprop(x,y)
            # accumulate the weights and biases
            biases = [nb + db for nb, db in zip(biases,delta_b)]
            weights = [nw + dw for nw, dw in zip(weights,delta_w)]
        
        # update network using gradient descent update rule
        self.biases = [b - (alpha/len(mini_batch))*nb 
                        for b, nb in zip(self.biases, biases)]
        self.weights = [(1 - (alpha*lmbda/n))*w - (alpha/len(mini_batch))*nw
                        for w,nw in zip(self.weights, weights)]
    
    def SGD(self,training_data,epochs,mini_batch_size,alpha,lmbda,evaluation_data):
        """Train the network using mini-batch stochastic gradient descent

        """
        n = len(training_data)
        n_data = len(evaluation_data)

        evaluation_cost = []
        evaluation_accuracy = []
        training_cost = []
        training_accuracy = []
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.gd_mini_batch(mini_batch,alpha,lmbda,n)
            logger.info("Epoch "+ str(i) +" training complete")
            # training cost and accuracy
            cost = self.total_cost(training_data,lmbda)
            training_cost.append(cost)
            logger.info("Cost on training data: "+str(cost))
            accuracy = self.accuracy(training_data)
            training_accuracy.append(accuracy)
            logger.info("Accuracy on training data: "+str(accuracy)+"/"+str(n))
            # evaluation cost and accuracy
            cost = self.total_cost(evaluation_data,lmbda)
            logger.info("Cost on evaluation data: "+str(cost))
            evaluation_cost.append(cost)
            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            logger.info("Accuracy on evaluation data: "+str(accuracy)+"/"+str(n_data))
        
        return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy

    def accuracy(self,data):
        """Returns the number of input in data for which neural network 
        outputs the correct result.
        """
        results = [(np.argmax(self.feed_forward(x)),np.argmax(y)) for(x, y) in data]
        return sum( int(x == y) for(x,y) in results)

    def total_cost(self,data,lmbda):
        """Return the total cost of the network for dataset
        """
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            cost += self.cost.fn(a,y)/len(data)
        # add regularization
        cost += 0.5*(lmbda/len(data))*sum( np.linalg.norm(w)**2 for w in self.weights )
        return cost

    def vector_result(self,j):
        """Convert output value into network output vector
        """
        vec = np.zeros((self.sizes[-1],1))
        vec[j] = 1.0
        return vec
    
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))