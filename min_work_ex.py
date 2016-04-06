import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d

minibatch = 3
image_height,image_width = 28,28
filter_height,filter_width = 3,3 
n_filters = 1
n_channels = 1

n = 1/(np.sqrt(image_height*image_width))
X = T.tensor4(name='X')
X_shape = (minibatch,n_channels,image_height,image_width)
W_shape = (n_filters,n_channels,filter_height,filter_width)
W = theano.shared(n*rng.randn(*W_shape),name='W')

conv_out = conv2d(X, 
                  W, 
                  input_shape=X_shape, 
                  filter_shape=W_shape, 
                  border_mode='valid')
f = theano.function([X],[conv_out])

X_data = np.array(rng.randint(low=0,high=256,size=X_shape))  
conv_out = f([X_data]) 

