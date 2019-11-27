import os;
import torch;
import numpy as np;
import h5py;
#
def norm_vec(ptsy,ptsx):
    xm = np.mean( ptsx , axis=0, keepdims = True );
    x = ptsx - xm;
    s = np.max( np.sum( np.sqrt( x**2 ), axis = 1,keepdims = True ),axis = 0,keepdims=True);
    y = 0.5 * ( ptsy - xm ) / s;
    return np.mean(y,axis = 0);
#
def ncv(h5fname):
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
                    if plst[pi].shape[0] > 3 and plst[pj].shape[0] > 3:
                        vec = None;
                        if plst[pi].shape[0] > plst[pj].shape[0]:
                            vec = norm_vec( plst[pj], plst[pi] );
                        else:
                            vec = norm_vec( plst[pi], plst[pj] );
                        if vec is not None:
                            v.append(vec);
    X = None;
    if len(v) > 0:
        X = np.stack(v);
    h5f.close();
    return X;