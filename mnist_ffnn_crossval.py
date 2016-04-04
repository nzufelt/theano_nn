"""
Performs cross validation to determine the best parameters for a ffnn on the
MNIST digits.

The parameters we test along:
    minibatch -- the size of the minibatch.  Note that in order for this
                     to be fair across minibatch sizes we need to tweak the
                     code so that each epoch runs through each element once
    epochs -- the number of full iterations through the code
    n_hidden -- the number of nodes in the hidden layer of our neural net
     

"""
