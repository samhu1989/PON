#coding:utf-8
from __future__ import print_function
from __future__ import division
#python import
import os;
import shutil;
import json;
import random;
import trimesh;
from .deal_partnet import seen_cat,unseen_cat,cat_size;
#
def dealobjs(path,opath):
    objply = path+os.sep+'objs.ply'
    if os.path.exists(objply):
        mesh = trimesh.load(objply);
    else:
        fs = os.listdir(path+os.sep+'objs');
        for i,f in enumerate(fs):
            if i == 0:
                mesh = trimesh.load(path+os.sep+'objs'+os.sep+f);
            else:
                mesh += trimesh.load(path+os.sep+'objs'+os.sep+f);
    T = trimesh.transformations.translation_matrix(-mesh.centroid);
    S = trimesh.transformations.scale_matrix(2.0 / mesh.scale,[0, 0, 0]);
    M = trimesh.transformations.concatenate_matrices(T, S)
    mesh.apply_transform(M);
    pts = mesh.sample(10000);
    pc = trimesh.Trimesh(vertices=pts);
    mesh.export(opath+'_mesh.ply');
    mesh.export(opath+'_mesh.obj');
    pc.export(opath+'_pts.ply');

def run(**kwargs):
    data_root = kwargs['data_path'];
    train = data_root + os.sep + 'train';
    if not os.path.exists(train):
        os.mkdir(train);
    val = data_root + os.sep + 'val';
    if not os.path.exists(val):
        os.mkdir(val);
    for cat in (seen_cat + ['augment']):
        cat_root = data_root+os.sep+cat;
        fs = os.listdir(cat_root);
        for i,f in enumerate(fs):
            print(cat,i,'/',len(fs));
            dealobjs(cat_root+os.sep+f,train+os.sep+cat+'_'+f);
    for cat in unseen_cat:
        cat_root = data_root+os.sep+cat;
        fs = os.listdir(cat_root);
        for i,f in enumerate(fs):
            print(cat,i,'/',len(fs));
            dealobjs(cat_root+os.sep+f,val+os.sep+cat+'_'+f);
    return ;