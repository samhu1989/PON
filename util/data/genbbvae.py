import os;
import time;
import torch;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import random;

cat2lbl = {};
#
def getX(h5fname):
    h5f = h5py.File(h5fname,'r');
    label = np.array(h5f['label']);
    pts = np.array(h5f['pts']);
    v = [];
    for i in range(label.shape[0]):
        lbl = label[i,:];
        plst = [];
        for l in range(np.min(lbl),np.max(lbl)):
            p = np.array( pts[i,lbl==l,:] );
            if p.size > 1:
                plst.append(p);
        #
        if len(plst) > 1:
            for pi in range(len(plst)-1):
                for pj in range(pi+1,len(plst)):
                    if plst[pi].shape[0] > plst[pj].shape[0]:
                        vec = np.mean(plst[pj],axis=0) - np.mean(plst[pi],axis=0);
                    else:
                        vec = np.mean(plst[pi],axis=0) - np.mean(plst[pj],axis=0);
                    v.append(vec);
    X = None;
    if len(v) > 0:
        X = np.stack(v);
    h5f.close();
    return X;
#center vector
def getdata(root,opt):
    cnt = 0;
    Xs = [];
    for root, dirs, files in os.walk(root, topdown=False):
       for name in files:
          f = os.path.join(root, name)
          if f.endswith('.h5'):
            print(os.path.basename(root));
            cat2lbl[os.path.basename(root)] = cnt;
            Xv = getX(f);
            Xs.append(Xv);
            cnt += 1;
    X = np.concatenate(Xs,axis=0);
    return X;

def run(**kwargs):
    assert kwargs['user_key'] , "please set user_key, current value is %s"%kwargs['user_key']; 
    root = kwargs['data_path'];
    print("getting data");
    X,labels = getdata(root,kwargs);
    print(X.shape);
    