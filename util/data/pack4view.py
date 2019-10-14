#coding:utf-8
from __future__ import print_function
from __future__ import division
#python import
import os;
import shutil;
import json;
import random;
import numpy as np;
#project import
from .ply import read_ply;
from ..tools import write_pts2sphere;

def pack_data(path,key):
    return;

def pack_res(path):
    packed_path = path + os.sep + 'packed';
    if not os.path.exists(packed_path):
        os.mkdir(packed_path);
    fs = os.listdir(path);
    for f in fs:
        if f.endswith('input.png'):
            batch = '_'+f.split('_')[1];
            if not os.path.exists(packed_path+os.sep+batch):
                os.mkdir(packed_path+os.sep+batch);
            src = path+os.sep+f;
            print(batch);
            print(f);
            tgt = packed_path+os.sep+batch+os.sep+f[len(batch):];
            print(src);
            print(tgt);
            try:
                shutil.copyfile(src,tgt);
                shutil.copyfile(src.replace('input.png','y.ply'),tgt.replace('input.png','y.ply'));
                shutil.copyfile(src.replace('input.png','ypt.ply'),tgt.replace('input.png','ypt.ply'));
                ply = read_ply(src.replace('input.png','gt.ply'));
                pts = np.array(ply['points']);
                write_pts2sphere(tgt.replace('input.png','gt.ply'),pts);
            except Exception as e:
                print(e);
                continue;

def run(**kwargs):
    data_root = kwargs['data_path'];
    if os.path.basename(data_root)=='ply':
        pack_res(data_root);
    elif os.path.basename(data_root)=='train':
        pack_data(data_root,'augment');
    else:
        pack_res(data_root);
    return;