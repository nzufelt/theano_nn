"""
Implement a convolutional neural network using theano.  This script was
created by Nicholas Zufelt as a part of the London Machine Learning
Practice meetup.
"""
import sys
import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import relu,conv2d
# New implementation is here:
# http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d
# OLD:
#from theano.tensor.nnet.conv import conv2d

#### Parameters
# For the whole network
reg,alpha = .01,.01 #(float(i) for i in sys.argv[4:6])
minibatch = 200 #(int(i) for i in sys.argv[1:4])
epochs,print_every = 100,10 #(int(i) for i in sys.argv[6:8])
# For the convolutional layer
image_height,image_width = 28,28
filter_height,filter_width = 3,3 
pool_size,n_filters = (2,2),1 # n_filters is the number of copies of the filter
# For the Fully-connected layer
n_inputs = ((image_height - filter_height + 1) // pool_size[0])**2
# figure out why above is correct! Note that we assume square everything
# for MNIST, with 3x3 filter and 2x2 pooling, this gives 13x13=169
n_hidden,n_outputs = 70,10

# Need to initialize the weights to a small, random number
n = 1/(np.sqrt(n_inputs))

# Input and training values
X = T.tensor4(name='X')
X_shape = (minibatch,1,image_height,image_width)  # 1 ==> greyscale
y = T.dmatrix('y') # one-hot vectors

# Convolution layer
W_shape = (n_filters,1,filter_height,filter_width) # 1's ==> greyscale
W_conv = theano.shared(n*rng.randn(*W_shape),
                       name='W_conv')
conv_out_layer = conv2d(X, W_conv,
                        input_shape=X_shape,
                        filter_shape=None,#W_shape,
                        border_mode='valid')
# Note:
# output_shape = (minibatch, 1, output_rows, output_columns)

# Pooling layer
pooled_out = pool_2d(input=conv_out_layer,
                     ds=pool_size,
                     ignore_border=True)
# ignore_border ==> round down if convolution_output / pool_size is not int
# OLD: pooled_out = T.signal.downsample.max_pool_2d(input=conv_out,

# Implement the bias term and nonlinearity
b_conv = theano.shared(np.zeros(n_filters,), name='b_conv')
conv_out = relu(pooled_out + b_conv.dimshuffle('x',0,'x','x'))

#TODO: check that the output of conv_out is the right dimension
#          reshape to being one-dimensional, stitch into z1
###### NOTE: this is bugged at the moment, conv2d is prepending a dimension to my X_data
conv_out_flat = conv_out.flatten()

f = theano.function([X],[conv_out_layer,conv_out,pooled_out])


"""DEBUG: disconnect the layers

# Fully-connected layers
W1_full = theano.shared(n*rng.randn(n_inputs,n_hidden), name='W1_full')
b1_full = theano.shared(np.zeros(n_hidden), name='b1_full')

W2_full = theano.shared(n*rng.randn(n_hidden,n_outputs), name='W2_full')
b2_full = theano.shared(np.zeros(n_outputs), name='b2_full')

z1 = conv_out_flat.dot(W1_full) + b1_full
hidden = relu(z1)
z2 = hidden.dot(W2_full) + b2_full
output = T.nnet.softmax(z2)
prediction = np.argmax(output,axis=1)
crossent = T.nnet.categorical_crossentropy(output,y)

cost = crossent.sum() + reg*((W1_full**2).sum()+(W2_full**2).sum())

# gradients and update statements
params = [W_conv,b_conv,W1_full,b1_full,W2_full,b2_full]
grads = T.grad(cost,[*params])
updates = tuple((param,param - alpha * grad) for param, grad in zip(params,grads))

# build theano functions
epoch = theano.function(inputs = [X,y],
                        outputs = [output, cost],
                        updates = updates)
predict = theano.function(inputs=[X],outputs=prediction)
"""
# Read in MNIST data
train_df = pd.read_csv('train.csv')[0:200]
X_data = train_df.values
del train_df # free up some memory
I = np.identity(10)
y_data = np.array([I[i] for i in X_data.T[0].T]) # one-hot the y's
# strip off response variable and make into a 4-tensor:
X_data = np.reshape(X_data.T[1:].T,(minibatch,1,image_height,image_width))  

print(X_data.shape)
cout_layer,cout,pout = f([X_data]) 


"""" DEBUG
# Give the data mean 0:
X_data = X_data.astype(float)
X_data -= 128.0
L = X_data.shape[0]

# train the model
for i in range(epochs):
    for ind in range(0,L,minibatch):
        rows = list(range(ind,min(ind+minibatch,L+1)))
        pred,err = epoch(X_data[rows,:],y_data[rows,:])
        if i % print_every == 0:
            print('Error after epoch {}: {}'.format(i,err))
"""
# TODO: some kind of accuracy testing
"""
OLD:
if n_outputs == 1:
    preds = predict(D[0]).T[0]
    wrong = (preds != D[1]).sum()
else:
    I = np.identity(n_outputs)
    preds = np.array([I[i] for i in predict(D[0])])
    wrong = (preds != D[1]).sum() / 2                      # note the /2

score = (N*1.0 - wrong)/N
print("Our model made {} errors, for an accuracy of {}".format(wrong, score))
"""
