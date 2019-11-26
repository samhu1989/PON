import os;
import time;
import torch;
import numpy as np;
import matplotlib.pyplot as plt;
from ..tsne import tsne;
import h5py;
import logging;
import random;
from matplotlib.colors import ListedColormap,BoundaryNorm;
#from mpl_toolkits.axes.grid1 import make_axes_locatable

cat2lbl = {};
#center vector
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
#

def getdata(root,opt):
    cnt = 0;
    Xs = [];
    Ls = [];
    for root, dirs, files in os.walk(root, topdown=False):
       for name in files:
          f = os.path.join(root, name)
          if f.endswith('.h5'):
            print(os.path.basename(root));
            cat2lbl[os.path.basename(root)] = cnt;
            Xv = eval(opt['user_key'])(f);
            if Xv is not None:
                if Xv.shape[0] > 100:
                    index = random.choices(range(Xv.shape[0]), k=100)
                    Xv = Xv[index,:];
                Xs.append(Xv);
                Lv = np.zeros(Xv.shape[0]);
                Lv.fill(cnt);
                Ls.append(Lv);
            cnt += 1;
    X = np.concatenate(Xs,axis=0);
    L = np.concatenate(Ls,axis=0);
    return X,L;

def run(**kwargs):
    assert kwargs['user_key'] , "please set user_key, current value is %s"%kwargs['user_key']; 
    root = kwargs['data_path'];
    print("getting data");
    X,labels = getdata(root,kwargs);
    print(X.shape);
    start_time = time.time();
    Y = tsne(X, 2, 50, 20.0, max_iter=0);
    Y = Y.cpu().numpy();
    #
    print("--- %s seconds ---" % (time.time() - start_time));
    tb20b = plt.cm.get_cmap('tab20b');
    tb20c = plt.cm.get_cmap('tab20c');
    colors = [tb20b.colors[i] for i in range(len(tb20b.colors))];
    colors.extend( [tb20c.colors[i] for i in range(4)] );
    cm = ListedColormap(colors)
    norm = BoundaryNorm(np.linspace(-0.5,24.5,25), cm.N);
    #data
    fig = plt.figure(figsize=(12.8, 9.6));
    ax = plt.subplot(121)
    #
    pts = ax.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap=cm,norm=norm);
    #legend
    cbar = plt.colorbar(pts,fraction=0.15, aspect=4,shrink=2.0)
    #
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(cat2lbl.keys()):
        cbar.ax.text(0.0 ,(j + 0.5) / 24.0 , lab, ha='left', va='center')
    #
    ax = plt.subplot(122)
    print('Chair:',cat2lbl['Chair']);
    print('Table:',cat2lbl['Table']);
    idx = np.logical_or( labels == cat2lbl['Chair'],labels == cat2lbl['Table'] );
    pts = ax.scatter(Y[idx, 0], Y[idx, 1], s=20, c=labels[idx], cmap=cm,norm=norm);
    #legend
    plt.savefig('./log/tsne.png');