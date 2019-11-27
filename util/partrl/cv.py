#
import os;
import torch;
import numpy as np;
import h5py;

def cv(h5fname):
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