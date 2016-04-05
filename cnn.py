"""
Implement a convolutional neural network using Theano. created
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
        minibatch -- the size of the minibatch used in SGD.  The default is 0
                         corresponding to standard GD.
        reg -- the regularization parameter
        alpha -- the learning rate
        n_epochs -- number of training epochs (iterations of gradient descent)
        print_every -- print the error after this many epochs
    """

    def __init__(self, n_inputs, n_hidden, n_outputs, n_epochs=4000, print_every = 500, reg=.01, alpha=.01, minibatch = 0):
        # Need to initialize the parameters to a small, random number
        n = 1/np.sqrt(n_inputs*n_hidden*n_outputs)
        self.n_inputs,self.n_hidden,self.n_outputs,self.reg,self.alpha=n_inputs,n_hidden,n_outputs,reg,alpha
        self.epochs = n_epochs
        self.print_every = print_every
        self.minibatch = minibatch

        # Part of the class so that they don't get tossed back and forth between methods
        X = T.dmatrix('X')
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

        # Feedforward
        z1 = X.dot(self.W1)+self.b1
        hidden = T.tanh(z1)
        z2 = hidden.dot(self.W2) + self.b2
        if self.n_outputs == 1:
            output = T.nnet.sigmoid(z2)
            prediction = output > 0.5
            crossent = -y.dot(T.log(output)) - (1-y).dot(T.log(1-output))
        else:
            output = T.nnet.softmax(z2)
            prediction = np.argmax(output,axis=1)
            crossent = T.nnet.categorical_crossentropy(output,y)

        cost = crossent.sum() + reg*((self.W1**2).sum()+(self.W2**2).sum())

        # gradients
        gW1,gb1,gW2,gb2 = T.grad(cost,[self.W1,self.b1,self.W2,self.b2])

        # build theano function for gradient descent
        self.epoch = theano.function(inputs = [X,y],
                                     outputs = [output, crossent.sum()],
                                     updates = ((self.W1,self.W1-self.alpha*gW1),
                                                (self.b1,self.b1-self.alpha*gb1),
                                                (self.W2,self.W2-self.alpha*gW2),
                                                (self.b2,self.b2-self.alpha*gb2)))
        self.predict = theano.function(inputs=[X],outputs=prediction)

    def fit(self,X,y):
        """ Fit the model.

        Params:
            X -- 2d np.array of training data, rows are samples, columns are
                    features
            y -- in binary classication: a 1d np.array of labels (0 or 1)
                 in multiclassification: a 2d np.array of one-hot labels
        """
        def run_epoch(X,y):
            pred,err = self.epoch(X,y) # Note: not self.X or self.y
            if i % self.print_every == 0:
                print('Error after epoch {}: {}'.format(i,err))

        if self.minibatch == 0:
            # performing vanilla gradient descent
            for i in range(self.epochs):
                run_epoch(X,y)
        else:
            # performing minibatch gradient descent
            L = len(X)
            for i in range(self.epochs):
                rows = list(np.random.randint(size=self.minibatch,low=0,high=L))
                run_epoch(X[rows,:],y[rows,:])


    def training_accuracy(self,X,y):
        """ Originally, we used the whole of X at once, but this led to
        memory overflow.  So now we compute it a bit at a time.
        """

        n_samples = len(X)
        total_wrong = 0
        if self.n_outputs == 1:
            for i in range(0,len(X),1000):
                preds = self.predict(X[i:i+1000,:]).T[0]
                total_wrong += (preds != y[i:i+1000]).sum()
        else:
            I = np.identity(self.n_outputs)
            for i in range(0,len(X),1000):
                preds = np.array([I[i] for i in self.predict(X[i:i+1000,:])])
                total_wrong = (preds != y[i:i+1000,:]).sum() / 2  # note the /2

        score = (n_samples*1.0 - total_wrong)/n_samples
        print("Our model made {} errors, for an accuracy of {}".format(total_wrong,
                                                                       score))

    def _generate_toy_data(self,n_samples):
        """ Used for debugging."""
        if self.n_outputs == 1:
            D = (np.random.randn(n_samples,self.n_inputs),
                 np.random.randint(size=n_samples,low=0,high=2))
        else:
            I = np.identity(self.n_outputs)
            D = [np.random.randn(n_samples,self.n_inputs)]
            D.append(np.array([I[i] for i in np.random.randint(size=n_samples,
                                                               low=0,
                                                               high=self.n_outputs)]))
        return D

    def check_on_toy_data(self,n_samples):
        """ Used for debugging."""
        D = self._generate_toy_data(n_samples)

        self.fit(D[0],D[1])

        # check accuracy
        if self.n_outputs == 1:
            preds = self.predict(D[0]).T[0]
            wrong = (preds != D[1]).sum()
        else:
            I = np.identity(self.n_outputs)
            preds = np.array([I[i] for i in self.predict(D[0])])
            wrong = (preds != D[1]).sum() / 2                      # note the /2

        score = (n_samples*1.0 - wrong)/n_samples
        print("Our model made {} errors, for an accuracy of {}".format(wrong, score))
