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
op = './data/cage/'
trainnum = 2000;
valnum = 100;
testnum = 100;
def run(**kwargs):
    spsub = os.listdir(sp);
    for sub in spsub:
        if os.path.isdir(os.path.join(sp,sub)) and (not '.json' in sub) :
            ids = os.listdir(os.path.join(sp,sub));
            for id in ids:
                shapenet[id] = os.path.join(sp,sub,id);
    cats = os.listdir(pn);
    for c in cats:
        if c == 'Chair':
            num = trainnum+valnum+testnum;
        else:
            num = testnum;
        if os.path.isdir(os.path.join(pn,c)):
            fs = os.listdir(os.path.join(pn,c));
        for f in fs:
            if 'train' in f and f.endswith('.h5'):
                print(f);
                h5f = h5py.File(os.path.join(pn,c,f));
                jf = json.load(open(os.path.join(pn,c,f.replace('.h5','.json')),'r'));
                for i in range(len(jf)):
                    label = h5f['label'][i,...];
                    pts = h5f['pts'][i,...];
                    print('pn',np.max(label) - np.min(label))
                    if np.max(label) - np.min(label) < 20:
                        mid = jf[i]['model_id'];
                        if mid in shapenet.keys():
                            print(c,num);
                            if c == 'Chair' and num > testnum:
                                oc = os.path.join(op,'train',c);
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
                            pnf = os.path.join(oc,mid,'pn.h5');
                            pnj = os.path.join(oc,mid,'pn.json');
                            json.dump(jf[i],open(pnj,'w'));
                            pnh5 = h5py.File(pnf,'w');
                            for k in h5f.keys():
                                pnh5.create_dataset(k,data=h5f[k][i,...]);
                            fply = os.path.join(oc,mid,'pn.ply');
                            write_ply(fply,points=pd.DataFrame(np.array(pts))); 
                            
                        if num <= 0:
                            break;
                    if num <= 0:
                        break;
                            
                            
    return;