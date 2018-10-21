import numpy as np

def relu(x):
    x[x < 0] = 0
    return x

def drelu(x):
    x[x>0]=1
    x[x<=0]=0
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def suqareloss(y,y_):
    m = len(y)
    return np.sum(1/2.*np.square(y-y_))/m

def dsquareloss(y,y_):
    return (y-y_)

## going to be completed

def softmax(x):

    return

def dsoftmax(x):

    return

def cross_entropy(x):

    return

def dcross_entropy(x):

    return