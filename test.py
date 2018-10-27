import numpy as np
from main import MLPnet

def test_on(img,lbs,w,im_len=6000):
    test_img = img[0:im_len,:]
    [_,_,_,out] = MLPnet(test_img,w)

    res = np.argmax(np.asarray(out),axis=1)
    comp = np.equal(lbs[0:im_len],res)

    stat = np.zeros(len(test_img))
    stat[comp==True]=1
    acc = np.sum(stat)/len(test_img)
    return acc
