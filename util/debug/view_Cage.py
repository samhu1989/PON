import traceback
import importlib
import sys;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from torch.utils.data import DataLoader;
from ..data.gen_toybox import box_face as bf;
from net.g.box import Box;
from util.dataset.ToyV import recon,mv,proj;
from util.tools import partial_restore;
from ..data.obb import OBB;
import os;
import h5py;
from PIL import Image;
from util.data.ply import read_ply,write_ply;
import pandas as pd;

def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);
                
def parse(vec):
    coord = np.array([[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]],dtype=np.float32);
    ss = vec[:3];
    coords = ss[np.newaxis,:]*coord;
    sr1 = vec[3:6];
    sr2 = vec[6:9]
    srot = rot(sr1,sr2);
    vs = np.dot(coords,srot.reshape(3,3))
    ts = vec[9:12];
    coordt = ts[np.newaxis,:]*coord;
    center = vec[12:15];
    tr1 = vec[15:18];
    tr2 = vec[18:21]
    trot = rot(tr1,tr2);
    vt = np.dot(coordt,trot.reshape(3,3)) + center[np.newaxis,:];
    return  vs,vt;
    
def write_box(box,path):
    obbp = [];
    obbf = [];
    for i in range(box.shape[0]):
        vec = box[i,...];
        obbp.append( OBB.v2points(vec) );
        obbf.append(bf + i*8);
    obbv = np.concatenate(obbp,axis=0);
    fidx = np.concatenate(obbf,axis=0);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[12*len(obbf)],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    write_ply(path,points=pd.DataFrame(obbv.astype(np.float32)),faces=pd.DataFrame(face));  
    
def getr3(r1,r2):
    rr1 = r1 / np.sqrt(np.sum(r1**2));
    rr2 = r2 - np.sum(r2*rr1)*rr1;
    rr2 = rr2 / np.sqrt(np.sum(rr2**2));
    r3 = np.cross(rr1,rr2);
    r3 = r3 / np.sqrt(np.sum(r3**2));
    return r3;

def add_box(i,touch,vec,box,lst,ilst):
    si = touch[i,0];
    ss = box[si,:3];
    sc = box[si,3:6]; 
    sr1 = box[si,6:9];
    sr2 = box[si,9:12];
    sr3 = getr3(sr1,sr2);
    sv = np.stack([ss,sc,sr1,sr2,sr3],axis=0).flatten();
    if not si in ilst:
        ilst.append(si);
        lst.append(sv);
    #
    ti = touch[i,1];
    ts = box[ti,0:3];
    tc = box[ti,3:6];
    if np.random.uniform(0.0,1.0) > 0.8:
        tr1 = box[ti,6:9];
        tr2 = box[ti,9:12];
    elif np.random.uniform(0.0,1.0) > 0.3:
        tr1 = vec[15:18];
        tr2 = box[ti,9:12];
    else:
        tr1 = vec[15:18];
        tr2 = vec[18:21];
    tr3 = getr3(tr1,tr2);
    #
    tv = np.stack([ts,tc,tr1,tr2,tr3],axis=0).flatten();
    if not ti in ilst:
        ilst.append(ti);
        lst.append(tv);
    
    
def infer_box(net,img,msk,touch,box):
    num = touch.shape[0];
    imgd = np.stack([img for x in range(num)],axis=0);
    imgd = torch.from_numpy(imgd);
    msksd = torch.from_numpy(msk[touch[:,0],:,:]);
    mskst = torch.from_numpy(msk[touch[:,1],:,:]);
    data = [imgd,msksd,mskst,None,None,None,None,None];
    data2cuda(data);
    net.eval();
    with torch.no_grad():
        out = net(data);
    y = out['y'].data.cpu().numpy();
    veco = out['vec'].data.cpu().numpy();
    obox = [];
    ibox = [];
    sorti = [i[0] for i in sorted(enumerate(y.tolist()), key=lambda x:x[1],reverse = True)]
    for i in sorti:
        add_box(i,touch,veco[i,...],box,obox,ibox);
    obox = np.stack(obox,axis=0);    
    return obox;


def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        for k in kwargs.keys():
            if not kwargs[k] is None:
                opt[k] = kwargs[k];
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
        if torch.cuda.is_available():
            net = net.cuda();
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+opt['dataset']);
        train_data = m.Data(opt,opt['user_key']);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
        
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
           
    if opt['model']!='':
        outdir = os.path.dirname(opt['model'])+os.sep+'view';
        if not os.path.exists(outdir):
            os.mkdir(outdir); 
    #
    root = os.path.join(opt['data_path'],'test');
    cat_lst = os.listdir(root);
    for c in cat_lst:
        path = os.path.join(root,c);
        cout = os.path.join(outdir,'_'+c);
        if not os.path.exists(cout):
            os.mkdir(cout);
        if os.path.isdir(path):
            f_lst = os.listdir(path);
            cnt = 0;
            for i,f in enumerate(f_lst):
                if f.endswith('.h5'):
                    fopath = os.path.join(cout,'_%04d'%cnt);
                    h5f = h5py.File(os.path.join(path,f),'r');
                    if not os.path.exists(fopath):
                        os.mkdir(fopath);
                    img = np.array(h5f['img']);
                    Image.fromarray((img*255.0).astype(np.uint8),mode='RGB').save(os.path.join(fopath,'_input.png'));
                    msk = np.array(h5f['msk']);
                    touch = np.array(h5f['touch']);
                    box = np.array(h5f['box']);
                    write_box(box,os.path.join(fopath,'_gt.ply'));
                    obox = infer_box(net,img,msk,touch,box);
                    write_box(obox,os.path.join(fopath,'_out.ply'));
                    h5f.close();
                    cnt += 1;
                if cnt > 10:
                    break;
            