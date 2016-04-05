"""
Implement a convolutional neural network using theano.

Calling this script, an example:
$ python

"""
import numpy as np
import numpy.random as rng
import sys
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d # New implementation
# OLD:
#from theano.tensor.nnet.conv import conv2d

# Receive inputs from user
n_inputs,n_outputs,n_hidden,minibatch = (int(i) for i in sys.argv[1:4])
reg,alpha = (float(i) for i in sys.argv[4:6])
epochs,print_every = (int(i) for i in sys.argv[6:8])
image_height,image_width,filter_height,filter_width,pool_size = [None]*5


# Need to initialize the weights to a small, random number
n = 1/(np.sqrt(n_inputs))

# Input and training values
X = T.tensor4(name='X')
X_shape = (minibatch,1,image_height,image_width)  # 1 ==> greyscale
y = T.dmatrix('y') # one-hot vectors

# Convolution layer
W_shape = (1,1,filter_height,filter_width) # 1's ==> greyscale
W_conv = theano.shared(n*rng.randn(*W_shape),
                       name='W_conv')
conv_out = conv2d(X, W_conv,
                    input_shape=X_shape,
                    filter_shape=W_shape,
                    border_mode='valid')
# Note:
# output_shape = (minibatch, 1, output_rows, output_columns)

# Pooling layer
pooled_out = T.signal.downsample.max_pool_2d(input=conv_out,
                                             ds=pool_size,
                                             ignore_border=True)
# Q: what does ignore_border do?

# TODO: implement the bias term
b_conv = theano.shared(np.zeros(), name='b_conv')
conv_out = T.relu(pooled_out + b_conv.dimshuffle())



# OLD:
#w_shape = (784,1,filter_height,filter_width)


# TODO: Fully-connected layer
W_full = theano.shared(n*np.random.randn(n_hidden,n_outputs), name='W2')
b_full = theano.shared(np.zeros(n_outputs), name='b2')

# TODO: probably delete all, but this is the stitching area
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

cost = crossent.sum() + reg*((W1**2).sum()+(W2**2).sum())


# gradients
gW1,gb1,gW2,gb2 = T.grad(cost,[W1,b1,W2,b2])

# build theano functions
epoch = theano.function(inputs = [x,y],
                        outputs = [output, crossent.sum()],
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
# train the model
for i in range(epochs):
    pred,err = epoch(D[0],D[1])
    if i % print_every == 0:
        print('Error after epoch {}: {}'.format(i,err))

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
