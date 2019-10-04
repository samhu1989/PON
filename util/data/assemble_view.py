import numpy as np;
import h5py;
from PIL import Image;
import pandas as pd;
import os;
from ..cmap import color;
from functools import partial;
from .ply import read_ply;
from ..tools import merge_mesh;

def map(x,idx):
    if (x == [0,0,0]).all():
        return color[idx,:];
    else:
        return x;
        
def run(**kwargs):
    data_root = kwargs['data_path'];
    res_root = kwargs['user_key'];
    ds = os.listdir(data_root);
    ds.sort();
    for idxd,d in enumerate(ds):
        idxw = idxd // 3;
        write_path = os.path.join("./log/as","_%03d_as"%idxw);
        read_path = os.path.join(data_root,d);
        if not os.path.exists(write_path):
            os.mkdir(write_path);
        fs = os.listdir(read_path);
        fs.sort();
        gt_merge = [];
        y_merge = [];
        for idxf,f in enumerate(fs):
            _f = os.path.join(read_path,f);
            if os.path.isfile(_f):
                img = Image.open(_f);
                imarr = np.array(img);
                tmp = imarr.reshape(-1,3);
                tmp = np.apply_along_axis( partial(map,idx=idxf), 1,tmp);
                tmp = tmp.reshape(224,224,3);
                fn = os.path.basename(_f);
                Image.fromarray(tmp).save(os.path.join(write_path,"_%03d_%03d.png"%(idxd,idxf)));
                #colorize the gt
                gt_fn = fn.replace('input.png','gt.ply');
                data = read_ply(os.path.join(res_root,gt_fn));
                gt_merge.append(data);
                #colorize the y
                y_fn = fn.replace('input.png','y.ply');
                data = read_ply(os.path.join(res_root,y_fn));
                y_merge.append(data);
        merge_mesh(os.path.join(write_path,"_%03d_gt.ply"%(idxd)),gt_merge);
        merge_mesh(os.path.join(write_path,"_%03d_y.ply"%(idxd)),y_merge,with_face=True);
                