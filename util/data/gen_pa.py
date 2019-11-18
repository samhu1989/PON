from __future__ import print_function
from PIL import Image
#sys import
import os;
import random;
import h5py;
import numpy as np;
from scipy import ndimage;
#torch import
import torch;
import torch.utils.data as data;
from ..data.ply import write_ply;
import pandas as pd;
import traceback;

def run(**kwargs):
    opt = kwargs;
    droot = opt['data_path'];
    datapath = [];
    for root, dirs, files in os.walk(droot, topdown=True):
        for fname in files:
            if fname.endswith('.h5'):
                datapath.append(os.path.join(root,fname));
    datapath.sort();
    fmap = [];
    smap = [];
    pmap = [];
    cat = [];
    import matplotlib.pyplot as plt;
    for idx,p in enumerate(datapath):
        #
        f = h5py.File(p,'r');
        cnt = f['cnt'];
        snum = cnt.shape[0];
        pnum = int(np.sum(cnt));
        #
        num = 0;
        
        for i in range(snum):
            img = f['img'][i+1,...];
            msk_lst = [];
            for j in range(int(cnt[i+1])):
                msk = f['msk'][j+num+1,...];
                msk_lst.append( msk );
                x = np.sum( msk > 0 );
            num += int(cnt[i+1]);
        #
            col = int(np.sqrt(len(msk_lst)));
            row = len(msk_lst) // col;
            plt.subplot(row,col,1);
            plt.imshow((img*255.0).astype(np.uint8));
            for ir in range(row):
                for ic in range(col):
                    ix = ir*col + ic + 1;
                    if ix == 1:
                        continue;
                    plt.subplot(row,col,ix);
                    if ix < len(msk_lst):
                        plt.imshow(msk_lst[ix]);
        f.close();
        #
    return;