import numpy as np
import matplotlib.pyplot as plt
from data_generator import data_loader
import nn

sigmoid = nn.sigmoid
loss = nn.suqareloss
dsigmoid = nn.dsigmoid
lossfunc = nn.suqareloss
dlossfunc = nn.dsquareloss

lr=0.005
epoch = 2000

def MLPnet(x,w1,b1,w2,b2,w3,b3,w4,b4):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = sigmoid(np.dot(y1, w2) + b2)
    y3 = sigmoid(np.dot(y2, w3) + b3)
    y4 = sigmoid(np.dot(y3, w4) + b4)
    return [y1,y2,y3,y4]

def relu(x):
    x[x < 0] = 0
    return x

def initweight(wsig=3,bsig=2):
    w1 = wsig*np.random.randn(2,30)
    b1 = bsig*np.ones([1,30])
    w2 = wsig*np.random.randn(30,30)
    b2 = bsig*np.ones([1,30])
    w3 = wsig*np.random.randn(30,40)
    b3 = bsig*np.ones([1,40])
    w4 = wsig*np.random.randn(40,4)
    b4 = bsig*np.ones([1,4])
    return np.asarray([w1,b1,w2,b2,w3,b3,w4,b4])

def onebackward(x,ylb,w1,b1,w2,b2,w3,b3,w4,b4):
    [y1, y2, y3, out] = MLPnet(x, w1, b1, w2, b2, w3, b3, w4, b4)
    dloss = dlossfunc(out, ylb)
    dlay4 = dsigmoid(np.dot(y3, w4))
    dw4 = np.dot(y3.T, dloss * dlay4)
    db4 = dlay4

    dlay3 = dsigmoid(np.dot(y2, w3))
    dw3 = np.dot(y2.T, dlay3)
    db3 = dlay3

    dlay2 = dsigmoid(np.dot(y1, w2))
    dw2 = np.dot(y1.T, dlay2)
    db2 = dlay2

    dlay1 = np.asarray([dsigmoid(np.dot(x, w1))])
    dw1 = np.dot(np.asarray([x]).T, dlay1)
    db1 = dlay1
    return np.asarray([dw1,db1,dw2,db2,dw3,db3,dw4,db4])

def train_one_pass(x,ylb,w1, b1, w2, b2, w3, b3, w4, b4):
    m = len(x)
    w = np.asarray([w1, b1, w2, b2, w3, b3, w4, b4])
    dw = np.zeros(w.shape)
    for i in range(m):
        dw = dw + onebackward(x[i], ylb[i], w1, b1, w2, b2, w3, b3, w4, b4)
    w =w - lr*dw/m
    return w

if __name__=='__main__':
    [w1, b1, w2, b2, w3, b3, w4, b4] = initweight()
    x_in,ylb = data_loader()
    w = np.asarray([w1, b1, w2, b2, w3, b3, w4, b4])
    losslist = []
    for i in range(epoch):
        w = train_one_pass(x_in,ylb, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
        [y1, y2, y3, out] = MLPnet(x_in, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
        loss = lossfunc(out, ylb)
        losslist.append(loss)
        if i%40==0:
            print('epoch is:',i,'present loss is:', np.sum(loss))
        if i%400==0 and (i>0):
            lr = lr/2.
            print('lr is changing:',lr)
    res1 = MLPnet(np.asarray([5, 5]), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
    res2 = MLPnet(np.asarray([5, 0]), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
    res3 = MLPnet(np.asarray([0, 0]), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])

    print('true label is:',1,'pred is:',np.argmax(res1[3])+1)
    print('true label is:',2,'pred is:',np.argmax(res2[3])+1)
    print('true label is:',3,'pred is:',np.argmax(res3[3])+1)

    res = MLPnet(x_in, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])
    trueres = np.argmax(ylb,axis=1)
    res = np.argmax(np.asarray(res[3]),axis=1)
    comp = np.equal(trueres,res)

    stat = np.zeros(len(x_in))
    stat[comp==True]=1
    acc = np.sum(stat)/len(x_in)
    print('the total accuracy is:',str(100*acc)+'%')

    plt.subplot(2,1,1)
    plt.scatter(x_in[:,0],x_in[:,1])
    plt.subplot(2,1,2)
    plt.plot(losslist)
    plt.show()