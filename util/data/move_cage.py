import numpy as np;
import h5py;
from PIL import Image;
import os;
from .obb import OBB;
from .gen_toybox import box_face as bf ;
from .ply import write_ply
import pandas as pd;
import scipy;
import shutil;


def run(**kwargs):
    root = kwargs['data_path'];
    out = kwargs['user_key'];
    cats = os.listdir(root);
    for c in cats:
        cpath = os.path.join(root,c);
        fs = os.listdir(cpath);
        copath = os.path.join(out,c);
        if not os.path.exists(copath):
            os.mkdir(copath);
        for f in fs:
            if f.endswith('.h5'):
                boxpath = os.path.join(cpath,f.rstrip('.h5'),'box.ply');
                if os.path.exists(boxpath):
                    src = os.path.join(cpath,f);
                    dst = os.path.join(copath,f);
                    outbox = os.path.join(copath,f.rstrip('.h5')+'.ply');
                    shutil.move(src,dst);
                    shutil.copy(boxpath,outbox);
                