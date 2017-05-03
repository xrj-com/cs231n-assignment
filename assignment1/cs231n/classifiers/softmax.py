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
  num_classes = W.shape[1]
  for i in xrange(num_train):
    X_W = X[i].dot(W)
    d_X_W = X[i]
    X_W -= np.max(X_W)
    X_W = np.exp(X_W)
    d_expX_W = d_X_W[:,np.newaxis].dot(X_W[np.newaxis,:])
    exp_sum = np.sum(X_W)
    loss -= np.log(X_W[y[i]]/exp_sum)
    dW_yi = np.zeros_like(dW)
    dW_yi[:,y[i]] = d_X_W
    dW += -dW_yi + d_expX_W/exp_sum

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W

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
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  exp_sum = np.sum(exp_scores, axis=1, keepdims=True)
  prob = exp_scores / exp_sum
  
  loss = -1.0/num_train * np.sum(np.log(prob[range(num_train), y])) + 0.5*reg*np.sum(W*W)

  prob[range(num_train), y] -= 1
  dW += X.T.dot(prob)
  dW /= num_train
  dW +=  reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

