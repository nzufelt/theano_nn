"""
A simple script to run a feedforward neural network on the MNIST digits.  

To run, call with the following hyperparameters:
    n_hidden -- the number of nodes in the hidden layer
    n_epochs -- the number of training epochs
    print_every -- print the error at this point
    reg -- the regularization parameter
    alpha -- the learning rate
    subset_train -- train on a subset of this size, for memory issues 
    subset_test -- test on a subset of this size, for memory issues

Such as:
$ python mnist_ffnn.py 625 10000 1000 .01 .01 4000 4000
"""
import sys
import numpy as np
import pandas as pd
from ffnn import ffnn

n_hidden,n_epochs,print_every = (int(i) for i in sys.argv[1:4])
reg,alpha = (float(i) for i in sys.argv[4:6])
subset_train,subset_test =(int(i) for i in sys.argv[6:]) 

# Read in training and test data
train_df = pd.read_csv('train.csv')[:subset_train]
X = train_df.values
del train_df # free up some memory
I = np.identity(10)

# Strip the labels off of the training data
y = [I[i] for i in X.T[0].T] 
X = X.T[1:].T 

# Give the data mean 0:
X = X.astype(float)
X -= 128.0

nn = ffnn(784,n_hidden,10,n_epochs,print_every,reg,alpha)
nn.fit(X,y)






# Do this later?
#X_test = pd.read_csv('test.csv').values
#nn.predict(X_test)






