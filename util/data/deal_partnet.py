from __future__ import print_function
from __future__ import division
#python import
import os;
import shutil;
import json;
#
seen_cat = ['Chair','StorageFurniture'];
unseen_cat = ['Table'];

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
        
def run(**kwargs):
    data_root = kwargs['data_path'];
    if not os.path.exists(data_root+os.sep+'pon'):
        os.mkdir(data_root+os.sep+'pon');
        select_cat(data_root+os.sep+'partnet',data_root+os.sep+'pon')
    return ;