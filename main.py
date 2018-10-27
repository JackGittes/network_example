import numpy as np
import matplotlib.pyplot as plt
from data_generator import data_loader
from data_generator import load_mnist
import nn
import params
import test

sigmoid = nn.sigmoid
dsigmoid = nn.dsigmoid
lossfunc = nn.squareloss
dlossfunc = nn.dsquareloss
softmax = nn.softmax
dsoftmax = nn.dsoftmax

def MLPnet(x,w):
    [w1,b1,w2,b2,w3,b3,w4,b4]=[w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]
    y1 = nn.relu(np.dot(x, w1) + b1)
    y2 = nn.relu(np.dot(y1, w2) + b2)
    y3 = nn.relu(np.dot(y2, w3) + b3)
    y4 = softmax(np.dot(y3, w4) + b4)
    return [y1,y2,y3,y4]

def initweight(wsig=0.01,bsig=0.1):
    w1 = wsig*np.random.randn(784,100)
    b1 = bsig*np.ones([1,100])
    w2 = wsig*np.random.randn(100,40)
    b2 = bsig*np.ones([1,40])
    w3 = wsig*np.random.randn(40,10)
    b3 = bsig*np.ones([1,10])
    w4 = wsig*np.random.randn(10,10)
    b4 = bsig*np.ones([1,10])
    return np.asarray([w1,b1,w2,b2,w3,b3,w4,b4])


def onebackward(x, ylb, w):
    [y1, y2, y3, out] = MLPnet(x, w)
    dloss = dlossfunc(out, ylb)
    dsoft = dsoftmax(np.dot(y3, w[6]) + w[7])
    dlay4 = np.dot(dloss,dsoft)
    dw4 = np.dot(y3.T, dlay4)
    db4 = dlay4

    dlay4_3 = np.dot(dlay4,w[6].T)
    dlay3 = dlay4_3*nn.drelu(np.dot(y2, w[4])+w[5])
    dw3 = np.dot(y2.T, dlay3)
    db3 = dlay3

    dlay3_2 = np.dot(dlay3,w[4].T)
    dlay2 = dlay3_2*nn.drelu(np.dot(y1, w[2])+w[3])
    dw2 = np.dot(y1.T, dlay2)
    db2 = dlay2

    dlay2_1 = np.dot(dlay2,w[2].T)
    dlay1 = dlay2_1*nn.drelu(np.dot(x, w[0])+w[1])
    dw1 = np.dot(np.asarray([x]).T, dlay1)
    db1 = dlay1
    return np.asarray([dw1,db1,dw2,db2,dw3,db3,dw4,db4])

def train_one_pass(x,ylb,w,lr=0.001):
    m = len(x)
    dw = np.zeros(w.shape)
    for i in range(m):
        dw = dw + onebackward(x[i], ylb[i], w)
    w =w - lr*dw/m
    return w

def train(lr=0.02,epochs=2000):
    w = initweight()
    x_in,ylb = data_loader()
    losslist = []
    for i in range(epochs):
        w = train_one_pass(x_in,ylb,w,lr)
        [_, _, _, out] = MLPnet(x_in,w)
        loss = lossfunc(out, ylb)
        losslist.append(loss)
        if i%40==0:
            print('epoch is:',i,'present loss is:', np.sum(loss))
    """
    calculate accuracy on the entire data
    """
    res = MLPnet(x_in, w)
    trueres = np.argmax(ylb,axis=1)
    res = np.argmax(np.asarray(res[3]),axis=1)
    comp = np.equal(trueres,res)

    stat = np.zeros(len(x_in))
    stat[comp==True]=1
    acc = np.sum(stat)/len(x_in)
    print('the total accuracy is:',str(100*acc)+'%')

    params.weight_writer(w)

    np.savetxt('loss.csv',np.asarray(losslist))
    plt.subplot(2,1,1)
    plt.scatter(x_in[:,0],x_in[:,1])
    plt.subplot(2,1,2)
    plt.plot(losslist)
    plt.show()

def prep_lab(lab):
    m = len(lab)
    tmp = np.zeros((m,10))
    for i in range(m):
        tmp[i,lab[i]]=1
    return tmp

def train_mnist(epoch=500,batch=256,lr=0.01):
    img,lab = load_mnist()
    w = initweight()
    losslist = []
    for k in range(epoch):
        for i in range(len(img)//batch):
            train_batch = img[i*batch:(i+1)*batch,:]
            train_lab = prep_lab(lab[i*batch:(i+1)*batch])
            w = train_one_pass(train_batch, train_lab, w, lr)
            [_,_,_,out] = MLPnet(train_batch,w)
            loss = lossfunc(out, train_lab)
            losslist.append(loss)
        print('current loss is:',loss)
        if k%20==0 and k>0:
            acc = test.test_on(img,lab,w)
            print('the total accuracy is:', str(100 * acc) + '%')
            params.weight_writer(w,weight_names='weight_'+str(k))
            np.savetxt('loss_'+str(k)+'.csv',np.asarray(losslist))

train_flag = 0
train_m = 1
if __name__ == "__main__":
    if train_flag==1 and train_m==0:
        train()
    elif train_flag==0 and train_m==1:
        train_mnist()