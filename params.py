import numpy as np
import json

def weight_loader(path='params/'):
    weight = []
    with open('params/weight','r') as fp:
        params = json.load(fp)
    for key,value in params.items():
        sp,data = value['shape'],value['data']
        tmp = np.asarray(data).reshape(sp)
        weight.append(tmp)
    return np.asarray(weight)

def weight_writer(w,path='params/',weight_names='weight'):
    key = 1
    params = {}
    for item in w:
        params.update({'w'+str(key):{'shape':item.shape,'data':list(item.flatten())}})
        key = key + 1
    with open(path+weight_names, 'w') as fp:
        json.dump(params,fp)