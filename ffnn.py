"""
Implement a feedforward neural network using theano
"""
import numpy as np
import theano
import theano.tensor as T
import pandas as pd

class ffnn(object):
    """ A standard (feedforward) neural network, with one hidden layer.

    Params:
        n_inputs -- number of nodes in the input layer
        n_outputs -- number of nodes in the output layer
        n_hidden -- number of nodes in the hidden layer
        reg -- the regularization parameter
        alpha -- the learning rate
        n_epochs -- number of training epochs (iterations of gradient descent)
    """

    def __init__(self, n_inputs, n_hidden, n_outputs, n_epochs=20000, reg=.01, alpha=.01):
        # Need to initialize the parameters to a small, random number
        n = 1/(np.sqrt(n_inputs * n_outputs * n_hidden))
        self.n_inputs,self.n_outputs,self.n_hidden,self.reg,self.alpha=n_inputs,n_outputs,n_hidden,reg,alpha
        self.epochs = n_epochs

        # Part of the class so that they don't get tossed back and forth between methods
        self.X = T.dmatrix('X')
        self.y = T.dmatrix('y') # should this be a vector?  I'm thinking a one-hotted vector

        # Weights and biases
        self.W1 = theano.shared(n*np.random.randn(n_inputs*n_hidden), name='W1')
        self.W2 = theano.shared(n*np.random.randn(n_hidden*n_outputs), name='W2')
        self.b1 = theano.shared(n*np.random.randn(n_inputs), name='b1')
        self.b2 = theano.shared(n*np.random.randn(n_hidden), name='b2')

        self.predict = None # will be a theano function

    def one_hot(self,y):
        """ Return a one-hot version of the numeral 0-9."""
        hot = np.zeros(self.n_outputs)
        hot[y] = 1
        return hot

    def cost(self):
        """ Compute the cost function for the current epoch.

        The cost function we will implement is regularized multi-class cross-entropy function, see below.
        """
        total_cost = 0.
        # Cost for the fit
        for x,label in zip(self.X,self.y):
            y_,_ = self.forward(x)
            # is this the right cost function?  It's different from the one used in NN-from-scratch
            # in that it penalyzes equally for false positives and negatives
            total_cost += (sum(-label*np.log(y_) - (1-y)*np.log(1-y_))/self.n_inputs)

        # Cost for the regularization
        total_cost += self.reg / (2*len(self.X)) * (sum(np.nditer(self.A**2)) + sum(np.nditer(self.B**2)))

        return total_cost

    def fit(X,Y):
        """ Fit the model.

        Params:
            X - 2d np.array of training data, rows are samples, columns are
                features
            Y - 1d np.array of training labels
        """

        x = T.matrix('x')
        y = T.dvector('y')

        # forward prop
        hidden = T.tanh(x*W1+b1)
        output = T.nnet.softmax(hidden*W2 + b2)
        prediction = output > 0.5

        # cost
        crossent = -y*T.log(output) - (1-y)*T.log(1-output)
        cost = crossent.mean() + self.reg * (w**2).sum()

        # gradients
        gW1,gb1,gW2,gb2 = T.grad(cost,[W1,b1,W2,b2])

        # build theano functions
        epoch = theano.function(inputs = [x,y],
                                outputs = [output, crossent],
                                updates = ((W1,W1-self.alpha*gW1),
                                           (b1,b1-self.alpha*gb1),
                                           (W2,W2-self.alpha*gW2),
                                           (b2,b2-self.alpha*gb2)))
        self.predict = theano.function(inputs=[x],outputs=prediction)

        # train the model
        for i in range(self.epochs):
            pred,err = epoch(X, Y) #note X, Y not x, y
            if i % print_every == 0:
                print('Error after epoch {}: {}'.format(i,err))
