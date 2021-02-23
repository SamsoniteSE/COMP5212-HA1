import numpy as np
import os
import math

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

def get_data():
    # Load datasets.
    train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
        dtype=float, delimiter=',') 
    test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
        dtype=float, delimiter=',') 
    train_x = train_data[:, :4]
    train_y = train_data[:, 4].astype(np.int64)
    test_x = test_data[:, :4]
    test_y = test_data[:, 4].astype(np.int64)

    return train_x, train_y, test_x, test_y

def compute_softmax_loss(W, X, y, reg):
    """
    Softmax loss function.
    Inputs:
    - W: D x K array of weight, where K is the number of classes.
    - X: N x D array of training data. Each row is a D-dimensional point.
    - y: 1-d array of shape (N, ) for the training labels.
    - reg: weight regularization coefficient.

    Returns:
    - softmax loss: NLL/N +  0.5 *reg* L2 regularization,
            
    - dW: the gradient for W.
    """
 

    #############################################################################
    # TODO: Compute the softmax loss and its gradient.                          #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = len(train_y)
    loss = 0
    dW = 0
    Xi = np.array([[0], [0], [0], [0]])
    for i in range(0, N):
        Xi[:,0] = X[i,:]
        for c in range(0, 3):
            if train_y[i] == 1:
                loss += W*Xi
    
    
    
    
    
    
    
    
    
    
    

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

reg = 0.0
W = [1, 2, 3, 6]
X, train_y, test_x, test_y = get_data()

N = len(train_y)
loss = [0.0]
dW = 0.0

for i in range(0, N):
    Xi = np.array([[X[i,0]], [X[i,1]], [X[i,2]], [X[i,3]]])
    for c in range(0, 3):
        if train_y[i] == 1:
            loss.append(float(np.matmul(W,Xi)))

print(Xi)
print(loss)
print(X[2,:])
print(np.transpose(X[2,:]))
print(len(train_y))
print(W)
print("success")