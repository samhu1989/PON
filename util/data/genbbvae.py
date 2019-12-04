import os;
import time;
import torch;
import numpy as np;
import matplotlib.pyplot as plt;
import h5py;
import random;
from .obb import OBB;
import scipy;
import scipy.spatial;

cat2lbl = {};
def obb2vec(obb):
    c = obb.centroid;
    ext = ( obb.max - obb.min ) / 2.0;
    rot = obb.rotation.reshape(-1);
    v = np.concatenate([c,ext,rot[:6]]);
    return v; 
#
def getX(h5fname):
    h5f = h5py.File(h5fname,'r');
    label = np.array(h5f['label']);
    pts = np.array(h5f['pts']);
    v = [];
    snum = min(label.shape[0],100);
    for i in range(snum):
        lbl = label[i,:];
        plst = [];
        for l in range(np.min(lbl),np.max(lbl)):
            p = np.array( pts[i,lbl==l,:] );
            if p.size > 1 and p.shape[0] > 5:
                plst.append(p);
        #
        if len(plst) > 1:
            num = min(20,len(plst));
            print(num);
            for pi in range(num-1):
                for pj in range(pi+1,num):
                    da,db = ( plst[pi],plst[pj] ) if plst[pi].shape[0] > plst[pj].shape[0] else ( plst[pj], plst[pi]);
                    tree = scipy.spatial.KDTree(da);
                    #dsta,idxa = tree.query(da,k=2);
                    dstb,idxb = tree.query(db,k=1);
                    #not connected
                    #if np.min(dstb) > np.mean(dsta[:,1]):
                    if np.min(dstb) > 0.02:
                        continue;
                    #
                    obba = OBB.build_from_points(da);
                    obbb = OBB.build_from_points(db);
                    va = obb2vec(obba);
                    vb = obb2vec(obbb);
                    v.append(np.concatenate([va,vb]))
                    v.append(np.concatenate([vb,va]))
        print(i,'/',snum);
        #
    X = None;
    if len(v) > 0:
        X = np.stack(v);
    h5f.close();
    return X;
#center vector
def getdata(root,opt):
    cnt = 0;
    out = opt['user_key']
    for root, dirs, files in os.walk(root, topdown=False):
       for name in files:
          f = os.path.join(root, name)
          if f.endswith('.h5'):
            catname = os.path.basename(root);
            if catname == 'Bag' or catname == 'Bottle' or catname == 'Bed':
                continue;
            cat2lbl[catname] = cnt;
            Xv = getX(f);
            if Xv is not None:
                h5f = h5py.File(os.path.join(out, catname+'.h5'),'w');
                dset = h5f.create_dataset("box_pair",data=Xv);
                h5f.flush();
                h5f.close();
            cnt += 1;
    return X;

def run(**kwargs):
    root = kwargs['data_path'];
    print("getting data");
    getdata(root,kwargs);
    