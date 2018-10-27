import numpy as np
import json
import struct
import os

def data_generator(data_len=600,catogories=4,sig=0.2):
    data_center = np.asarray([[15,15],[15,0],[0,0],[0,15]])
    x = sig*np.random.randn(data_len,2)
    num = data_len//catogories
    [ini,stp]=[0,num]
    ylb = np.zeros((data_len,catogories))
    for i in range(catogories):
        x[ini:stp,:]=x[ini:stp,:]+data_center[i]
        ylb[ini:stp,i] = 1
        [ini,stp] = [ini+num,stp+num]
    return x,ylb

def data_loader(path='data/'):
    with open(path + 'xin','r') as fp:
        data = json.load(fp)
    with open(path + 'labels', 'r') as fp:
        lbs = json.load(fp)
    [dsp,lbsp] = [data['shape'],lbs['shape']]
    [dlist,lblist] = [data['data'],lbs['lbs']]
    xin = np.asarray(dlist).reshape(dsp)
    labels = np.asarray(lblist).reshape(lbsp)
    return xin,labels

def load_mnist(path='MNIST/train', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

if __name__ == '__main__':
    x,ylb = data_generator()
    data = {'shape':x.shape,'data':list(x.flatten())}
    labels = {'shape':ylb.shape,'lbs':list(ylb.flatten())}
    with open('data/xin','w') as fp:
        json.dump(data,fp)
    with open('data/labels','w') as fp:
        json.dump(labels,fp)