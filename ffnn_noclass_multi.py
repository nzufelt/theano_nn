"""
Implement a feedforward neural network using theano.
"""
import numpy as np
import sys
import theano
import theano.tensor as T

# Receive inputs from user
n_inputs,n_outputs,n_hidden = (int(i) for i in sys.argv[1:4])
reg,alpha = (float(i) for i in sys.argv[4:6])
epochs,print_every = (int(i) for i in sys.argv[6:8])
run_test = bool(sys.argv[8])
toy_data_size = int(sys.argv[9])
# old, used to debug
#n_inputs,n_outputs,n_hidden,reg,alpha,epochs,print_every,run_test_size = 3,1,4,.01,.01,10000,1000,true,23

# Need to initialize the parameters to a small, random number
n = 1/(np.sqrt(n_inputs * n_outputs * n_hidden))

# Weights and biases
W1 = theano.shared(n*np.random.randn(n_inputs,n_hidden), name='W1')
W2 = theano.shared(n*np.random.randn(n_hidden,n_outputs), name='W2')
b1 = theano.shared(np.zeros(n_hidden), name='b1')
b2 = theano.shared(np.zeros(n_outputs), name='b2')

x = T.dmatrix('x')
if n_outputs == 1:
    # We are performing 2-class classification 
    y = T.dvector('y')
else:
    # We are performing multiclass (i.e. > 2) classification
    y = T.dmatrix('y')

# forward prop
z1 = x.dot(W1)+b1
hidden = T.tanh(z1)
z2 = hidden.dot(W2) + b2
if n_outputs == 1:
    output = T.nnet.sigmoid(z2)
    prediction = output > 0.5
    crossent = -y.dot(T.log(output)) - (1-y).dot(T.log(1-output))
else:
    output = T.nnet.softmax(z2)
    prediction = np.argmax(output,axis=1)
    crossent = T.nnet.categorical_crossentropy(output,y)

cost = crossent.sum() * ((W1**2).sum()+(W2**2).sum())

# gradients
gW1,gb1,gW2,gb2 = T.grad(cost,[W1,b1,W2,b2])

# build theano functions
epoch = theano.function(inputs = [x,y],
                        outputs = [output, crossent],
                        updates = ((W1,W1-alpha*gW1),
                                   (b1,b1-alpha*gb1),
                                   (W2,W2-alpha*gW2),
                                   (b2,b2-alpha*gb2)))
predict = theano.function(inputs=[x],outputs=prediction)

# generate toy data
N = toy_data_size
if n_outputs == 1:
    D = (np.random.randn(N,n_inputs),np.random.randint(size=N,low=0,high=2))
else:
    I = np.identity(n_outputs)
    D = [np.random.randn(N,n_inputs)]
    D.append(np.array([I[i] for i in np.random.randint(size=N,
                                                       low=0,
                                                       high=n_outputs)]))

#### TODO: build a predictor and training error calculator, fix crossentropy bug

# train the model
for i in range(epochs):
    pred,err = epoch(D[0],D[1])
    if i % print_every == 0:
        print('Error after epoch {}: {}'.format(i,err))
