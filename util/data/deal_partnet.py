#coding:utf-8
from __future__ import print_function
from __future__ import division
#python import
import os;
import shutil;
import json;
import random;
#
seen_cat = ['Chair','StorageFurniture'];
unseen_cat = ['Table'];
cat_size = 128;

def select_from_partnet(cpath,ctpath,dpath):
    shutil.copytree(cpath,ctpath);
    files = os.listdir(ctpath);
    for f in files:
        if f.endswith('.json'):
            js = json.load(ctpath+os.sep+f);
            ids = [ item.get('anno_id', 'NA') for item in js['items']];
            for id in ids:
                os.mkdir(ctpath+os.sep+id);
                try:
                    shutil.copytree(dpath+os.sep+id,ctpath+os.sep+id);
                except Exception as e:
                    print(e);

def select_cat(spath,tpath):
    if not os.path.exists(tpath+os.sep+'seen'):
        os.mkdir(tpath+os.sep+'seen');
    if not os.path.exists(tpath+os.sep+'unseen'):
        os.mkdir(tpath+os.sep+'unseen');
    for idx , cat in enumerate( seen_cat + unseen_cat ):
        cpath = spath+os.sep+'ins_seg_h5'+os.sep+'cat';
        dpath = spath+os.sep+'data_v0';
        ctpath = None;
        if idx < len(seen_cat):
            ctpath = tpath+os.sep+'seen'+os.sep+cat;
        else:
            ctpath = tpath+os.sep+'unseen'+os.sep+cat;
        if not os.path.exists(ctpath):
            os.mkdir(ctpath);
        select_from_partnet(cpath,dpath,ctpath);

def select_sample(spath,num):
    path = spath+os.sep+'pon_3_'+str(num);
    if not os.path.exists(path):
        os.mkdir(path);
        for cat in seen_cat + unseen_cat:
            os.mkdir(path+os.sep+cat);
        os.mkdir(path+os.sep+'augment');
    random.seed(100);
    for idx , cat in enumerate(seen_cat + unseen_cat):
        if idx < len(seen_cat):
            cat_source = spath+os.sep+'seen'+os.sep+cat;
            cat_target = path+os.sep+cat;
        else:
            cat_source = spath+os.sep+'unseen'+os.sep+cat;
            cat_target = path+os.sep+cat;
        flst = os.listdir(cat_source);
        random.shuffle(flst);
        cnt = num;
        for f in flst:
            if os.path.isdir(cat_source+os.sep+f):
                print(cat,cnt);
                try:
                    os.mkdir(cat_target+os.sep+f)
                    shutil.copytree(cat_source+os.sep+f+os.sep+'objs',cat_target+os.sep+f+os.sep+'objs');
                    shutil.copyfile(cat_source+os.sep+f+os.sep+'result_after_merging.json',cat_target+os.sep+f+os.sep+'result_after_merging.json')
                    shutil.copyfile(cat_source+os.sep+f+os.sep+'point_sample'+os.sep+'sample-points-all-pts-label-10000.ply',cat_target+os.sep+f+os.sep+'sample-points-all-pts-label-10000.ply')
                    shutil.copyfile(cat_source+os.sep+f+os.sep+'point_sample'+os.sep+'sample-points-all-pts-nor-rgba-10000.ply',cat_target+os.sep+f+os.sep+'sample-points-all-pts-nor-rgba-10000.ply')
                except Exception as e:
                    print(e);
            else:
                continue;
            cnt -= 1;
            if cnt <= 0:
                break;
        
def run(**kwargs):
    data_root = kwargs['data_path'];
    if not os.path.exists(data_root+os.sep+'pon'):
        os.mkdir(data_root+os.sep+'pon');
        select_cat(data_root+os.sep+'partnet',data_root+os.sep+'pon');
    select_sample(data_root+os.sep+'pon',cat_size)
    return ;