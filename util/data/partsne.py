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
from util.partrl import *;

cat2lbl = {};
#center vector
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
    for per in [5.0,30.0,50.0,100.0]:
        start_time = time.time();
        Y = tsne(X, 2, perplexity=per);
        Y = Y.cpu().numpy();
        #
        print("--- %s seconds ---" % (time.time() - start_time));
        tb20b = plt.cm.get_cmap('tab20b');
        tb20c = plt.cm.get_cmap('tab20c');
        colors = [tb20b.colors[i] for i in range(len(tb20b.colors))];
        colors.extend( [tb20c.colors[i] for i in range(4)] );
        cm = ListedColormap(colors)
        norm = BoundaryNorm(np.linspace(-0.5,23.5,25),cm.N);
        #data
        fig = plt.figure(figsize=(18.0, 6.0));
        ax1 = plt.subplot(131)
        ax1.axis('equal')
        #
        pts = ax1.scatter(Y[:, 0], Y[:, 1], s=20, c=labels, cmap=cm,norm=norm);
        ax1.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim());
        #legend
        ax2 = plt.subplot(132)
        cbar = plt.colorbar(pts,ax=ax2,fraction=1.0, aspect=1.0,shrink=1.0)
        #
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(cat2lbl.keys()):
            cbar.ax.text(0.0 ,(j + 0.5) / 24.0 , lab, ha='left', va='center')
        #
        ax3 = plt.subplot(133)
        ax3.axis('equal');
        ax3.set(xlim=ax1.get_xlim(),ylim=ax1.get_ylim());
        idx1 = np.logical_or( labels == cat2lbl['Chair'], labels == cat2lbl['Table'] );
        idx2 = np.logical_or( labels == cat2lbl['Knife'], labels == cat2lbl['Display'] );
        idx = np.logical_or(idx1,idx2);
        pts = ax3.scatter(Y[idx, 0], Y[idx, 1], s=20, c=labels[idx], cmap=cm,norm=norm);
        #legend
        plt.savefig('./log/tsne/tsne_%s_%f.png'%(kwargs['user_key'],per));
        plt.close(fig);