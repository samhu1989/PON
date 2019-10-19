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


def pack_res(path):
    packed_path = path + os.sep + 'packed';
    if not os.path.exists(packed_path):
        os.mkdir(packed_path);
    fs = os.listdir(path);
    for f in fs:
        if f.endswith('.png'):
            batch = '_'+f.split('_')[1];
            if not os.path.exists(packed_path+os.sep+batch):
                os.mkdir(packed_path+os.sep+batch);
            src = path+os.sep+f;
            tgt = packed_path+os.sep+batch+os.sep+f[len(batch):];
            shutil.copyfile(src,tgt);
            
    for f in fs:
        if ( f.endswith('.png') and (not f.endswith('input.png')) ) or f.endswith('.ply'):
            batch = '_'+f.split('_')[1];
            src = path+os.sep+f;
            tgt = packed_path+os.sep+batch+os.sep+f[len(batch):];
            shutil.copyfile(src,tgt);

def run(**kwargs):
    data_root = kwargs['data_path'];
    pack_res(data_root);
    return;