import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_features = W.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores) 
        p = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(p[y[i]])
        p[y[i]] = p[y[i]]-1
        dW +=  np.reshape(X[i], (num_features,1)).dot(np.reshape( p,(1,num_classes)))
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2*reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores) 
    p = np.exp(scores) / np.matmul(np.sum(np.exp(scores),axis=1).reshape((num_train,1)),np.ones((1,num_classes)))
    ind = list(range(num_train))
    loss = np.sum(-np.log(p[ind,y]))

    new_y = np.zeros(p.shape)
    new_y[ind,y]=1
    p = p - new_y
    dW = np.matmul(np.transpose(X),p)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2*reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

