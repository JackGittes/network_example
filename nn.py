########################################################################
#   Author : Zhao Mingxin
#   Start Date : 2018/10/21
#   Last Change : See Github Log
#   Homepage : jackgittes.github.io
########################################################################
'''
Description for this module :
define activation functions and their gradients
define loss function and their gradients
present version includes activations and lossfuncs as below:
    activations : sigmoid softmax relu
    lossfunc : square loss(hinge loss)
                cross entropy
the function list is increasing.
'''
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

def squareloss(y,y_):
    m = len(y)
    return np.sum(1/2.*np.square(y-y_))/m

def dsquareloss(y,y_):
    return (y-y_)

## going to be completed
def softmax(x):
    sp = x.shape
    if len(sp)==1:
        return np.exp(x)/np.sum(np.exp(x))
    else:
        return np.exp(x)/np.sum(np.exp(x),axis=1).reshape((len(x),1))

def dsoftmax(x):
 #   m = len(x)
    prob = softmax(x.flatten())
    return np.diag(prob.flatten())-np.dot(np.asarray([prob]).T,np.asarray([prob]))

def cross_entropy(y,y_):
    m = len(y)
    return -np.sum(y_*np.log(y)+(1-y_)*np.log((1-y)))/m

def dcross_entropy(y,y_):
    return -(y_*(1/y)-(1-y_)*(1/(1-y)))