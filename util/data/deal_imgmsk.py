import numpy as np;
import h5py;
from PIL import Image;
import pandas as pd;



def run(**kwargs):
    data_root = kwargs['data_path'];
    gname = kwargs['user_key'];
    ds = os.listdir(data_root);
    ds.sort();
    for idxd,d in enumerate(ds):
        idxw = idxd // 3;
        write_path = os.path.join("./log/as","_%03d_as"%idxw);
        read_path = os.path.join(data_root,idxd);
        if not os.path.exist(write_path):
            os.mkdir(write_path);
        fs = os.listdir(read_path);
        fs.sort();
        for idxf,f in enumerate(fs):
            _f = os.path.join(read_path,f);
            if os.path.isfile(_f):
                img = Image.open(_f);
                print(img.shape);
        
            
        