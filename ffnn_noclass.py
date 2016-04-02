"""
Implement a feedforward neural network using theano.
"""
import numpy as np
import theano
import theano.tensor as T

# Need to initialize the parameters to a small, random number
n_inputs,n_outputs,n_hidden,reg,alpha,epochs,print_every = 3,1,4,.01,.01,200,10
n = 1/(np.sqrt(n_inputs * n_outputs * n_hidden))

# Weights and biases
W1 = theano.shared(n*np.random.randn(n_inputs,n_hidden), name='W1')
W2 = theano.shared(n*np.random.randn(n_hidden,n_outputs), name='W2')
b1 = theano.shared(n*np.random.randn(n_hidden), name='b1')
b2 = theano.shared(n*np.random.randn(n_outputs), name='b2')

x = T.dmatrix('x')
y = T.dvector('y')

# forward prop
hidden = T.tanh(x.dot(W1)+b1)
output = T.nnet.softmax(hidden.dot(W2) + b2)
prediction = output > 0.5

# cost
crossent = -y*T.log(output) - (1-y)*T.log(1-output)
cost = crossent.mean() + reg * ((W1**2).sum()+(W2**2).sum())

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
N = 23 # number of samples, make it not divisible by 3 or 4 to ensure good broadcasting
D = (np.random.randn(N,n_inputs),np.random.randint(size=N,low=0,high=2))
print(D) 

# train the model
for i in range(epochs):
    pred,err = epoch(D[0],D[1])
    if i % print_every == 0:
        print('Error after epoch {}: {}'.format(i,err))
