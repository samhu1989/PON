import numpy as np;
import os;
import json;
import h5py;
import shutil;
from .ply import write_ply;
import pandas as pd;
sp = './data/ShapeNetCore.v2/'
shapenet = {}
pn = './data/ins_seg_h5/'
op = './data/cagenew/'
trainnum = 4000;
valnum = 200;
testnum = 200;
def run(**kwargs):
    spsub = os.listdir(sp);
    for sub in spsub:
        if os.path.isdir(os.path.join(sp,sub)) and (not '.json' in sub) :
            ids = os.listdir(os.path.join(sp,sub));
            for id in ids:
                shapenet[id] = os.path.join(sp,sub,id);
    with open(os.path.join(op,'selected.txt'),'w') as idlst:
        cats = os.listdir(pn);
        for c in cats:
            if c == 'Chair':
                num = trainnum+valnum+testnum;
            else:
                num = testnum;
            if os.path.isdir(os.path.join(pn,c)):
                fs = os.listdir(os.path.join(pn,c));
            for f in fs:
                if (('train' in f) or ('val' in f)) and f.endswith('.h5'):
                    print(f);
                    h5f = h5py.File(os.path.join(pn,c,f),'r');
                    print(h5f.keys());
                    jf = json.load(open(os.path.join(pn,c,f.replace('.h5','.json')),'r'));
                    for i in range(len(jf)):
                        label = h5f['label'][i,...];
                        pts = h5f['pts'][i,...];
                        print('pn',np.unique(label).size);
                        if np.unique(label).size < 20:
                            mid = jf[i]['model_id'];
                            print(mid,file=idlst);
                            if mid in shapenet.keys():
                                print(c,num);
                                if c == 'Chair' and num > testnum+valnum:
                                    oc = os.path.join(op,'train',c);
                                elif c == 'Chair' and num > testnum:
                                    oc = os.path.join(op,'val',c);
                                else:
                                    oc = os.path.join(op,'test',c);
                                if not os.path.exists(oc):
                                    os.mkdir(oc);
                                if not os.path.exists(os.path.join(oc,mid)):
                                    num -= 1;
                                    os.mkdir(os.path.join(oc,mid));
                                else:
                                    num -= 1;
                                    continue;
                                src = shapenet[mid];
                                dst = os.path.join(oc,mid,'sp')
                                shutil.copytree(src,dst);
                                
                            if num <= 0:
                                break;
                        if num <= 0:
                            break;
                    h5f.close();
        return;