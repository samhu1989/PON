import h5py;
import os;
import numpy as np;
from ..data.ply import write_ply;
import pandas as pd;
from PIL import Image;

def run(**kwargs):
    data_root = kwargs['data_path'];
    res_root = kwargs['user_key'];
    num = kwargs['batch_size'];
    datapath = [];
    for root, dirs, files in os.walk(data_root, topdown=True):
            for fname in files:
                if fname.endswith('.h5'):
                    datapath.append(os.path.join(root,fname));
    datapath.sort();
    for idx,p in enumerate(datapath):
        f = h5py.File(p,'r');
        cnt = f['cnt'];
        img = f['img'];
        msk = f['msk'];
        pc = f['pts'];
        for i in range(cnt.shape[0]):
            if int(cnt[i,:])==0:
                continue;
            cnum = cnt[i,...];
            start = int(np.sum(cnt[0:i,0]))+1;
            pobj = None;
            for ic in range(int(cnum)):
                if pobj is None:
                    pobj = pc[start+ic,...].copy();
                else:
                    pobj = np.concatenate([pobj,pc[start+ic,...].copy()],axis=0);
            imgf = Image.fromarray(np.uint8(255.0*img[i,...].copy()));
            imgf.save('./log/input%d.png'%num);
            write_ply('./log/output%d.ply'%num,points=pd.DataFrame(pobj),as_text=True);
            num -= 1;
            if num <= 0:
                exit();

            