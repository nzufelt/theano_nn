"""
Implement a feedforward neural network using theano
"""
import numpy as np
import theano
import theano.tensor as T

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

    def __init__(self, n_inputs, n_hidden, n_outputs, n_epochs=10000, print_every = 1000, reg=.01, alpha=.01):
        # Need to initialize the parameters to a small, random number
        n = 1/np.sqrt(n_inputs)
        self.n_inputs,self.n_hidden,self.n_outputs,self.reg,self.alpha=n_inputs,n_hidden,n_outputs,reg,alpha
        self.epochs = n_epochs

        # Part of the class so that they don't get tossed back and forth between methods
        self.X = T.dmatrix('X')
        if n_outputs == 1:
            # We are performing 2-class classification
            y = T.dvector('y')
        else:
            # We are performing multiclass (i.e. > 2) classification
            y = T.dmatrix('y')

        # Weights and biases
        self.W1 = theano.shared(n*np.random.randn(n_inputs,n_hidden), name='W1')
        self.W2 = theano.shared(n*np.random.randn(n_hidden,n_outputs), name='W2')
        self.b1 = theano.shared(np.zeros(n_hidden), name='b1')
        self.b2 = theano.shared(np.zeros(n_outputs), name='b2')

        """ With selfs?
        self.z1 = self.x.dot(self.W1)+self.b1
        self.hidden = T.tanh(self.z1)
        self.z2 = self.hidden.dot(self.W2) + self.b2
        if self.n_outputs == 1:
            self.output = T.nnet.sigmoid(self.z2)
            self.prediction = self.output > 0.5
            self.crossent = -self.y.dot(T.log(self.output)) - (1-self.y).dot(T.log(1-self.output))
        else:
            self.output = T.nnet.softmax(self.z2)
            self.prediction = np.argmax(self.output,axis=1)
            self.crossent = T.nnet.categorical_crossentropy(self.output,self.y)

        self.cost = self.crossent.sum() + reg*((self.W1**2).sum()+(self.W2**2).sum())
        """
        z1 = self.X.dot(self.W1)+self.b1
        hidden = T.tanh(z1)
        z2 = hidden.dot(self.W2) + self.b2
        if self.n_outputs == 1:
            output = T.nnet.sigmoid(z2)
            srediction = output > 0.5
            crossent = -self.y.dot(T.log(output)) - (1-self.y).dot(T.log(1-output))
        else:
            output = T.nnet.softmax(z2)
            prediction = np.argmax(output,axis=1)
            crossent = T.nnet.categorical_crossentropy(output,self.y)

        cost = crossent.sum() + reg*((self.W1**2).sum()+(self.W2**2).sum())

        # gradients
        gW1,gb1,gW2,gb2 = T.grad(cost,[W1,b1,W2,b2])

        # build theano function
        self.epoch = theano.function(inputs = [self.X,self.y],
                                     outputs = [output, crossent.sum()],
                                     updates = ((self.W1,self.W1-self.alpha*gW1),
                                                (self.b1,self.b1-self.alpha*gb1),
                                                (self.W2,self.W2-self.alpha*gW2),
                                                (self.b2,self.b2-self.alpha*gb2)))
        self.predict = theano.function(inputs=[self.X],outputs=prediction)

    def fit(self,X,y):
        """ Fit the model.

        Params:
            X -- 2d np.array of training data, rows are samples, columns are
                    features
            y -- in binary classication: a 1d np.array of labels (0 or 1)
                 in multiclassification: a 2d np.array of one-hot labels
        """
        for i in range(self.epochs):
            pred,err = self.epoch(X,y) # Note: not self.X or self.y
            if i % self.print_every == 0:
                print('Error after epoch {}: {}'.format(i,err))

    def check_on_toy_data(self,n_samples):
        """ Used for debugging."""
        # generate toy data
        N = n_samples
        if self.n_outputs == 1:
            D = (np.random.randn(N,self.n_inputs),
                 np.random.randint(size=N,low=0,high=2))
        else:
            I = np.identity(self.n_outputs)
            D = [np.random.randn(N,self.n_inputs)]
            D.append(np.array([I[i] for i in np.random.randint(size=N,
                                                               low=0,
                                                               high=self.n_outputs)]))
        # check accuracy
        if n_outputs == 1:
            preds = predict(D[0]).T[0]
            wrong = (preds != D[1]).sum()
        else:
            I = np.identity(n_outputs)
            preds = np.array([I[i] for i in predict(D[0])])
            wrong = (preds != D[1]).sum() / 2                      # note the /2

        score = (N*1.0 - wrong)/N
        print("Our model made {} errors, for an accuracy of {}".format(wrong, score))
