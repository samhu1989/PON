import traceback
import importlib
import sys;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation;
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from torch.utils.data import DataLoader;
from ..data.gen_toybox import box_face;
from net.g.box import Box;
from util.dataset.ToyV import recon;
import os;
from PIL import Image;
from util.data.ply import write_ply;
import pandas as pd;
from util.tools import partial_restore;
import h5py;
import heapq;
from net.cageutil import rot9np,normalize,rot6d,rot9;
import torch.nn.functional as F;

red_box = np.array(
    [
     [255,0,0],[255,0,0],[255,0,0],[255,0,0],
     [255,0,0],[255,0,0],[255,0,0],[255,0,0]
    ],dtype=np.uint8);
blue_box = np.array(
    [
     [0,0,255],[0,0,255],[0,0,255],[0,0,255],
     [0,0,255],[0,0,255],[0,0,255],[0,0,255]
    ],dtype=np.uint8);


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);
            
def parsegt(vec):
    vec = vec.copy();
    coord = np.array([[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]],dtype=np.float32);
    ss = vec[:3];
    coords = ss[np.newaxis,:]*coord;
    center = vec[3:6];
    srot = vec[6:15];
    box = np.dot(coords,srot.reshape(3,3)) + center[np.newaxis,:];
    return  box,center[np.newaxis,:];
    
def writebox(path,box,colors=None):
    fidx = box_face;
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    bn = len(box);
    face = np.zeros(shape=[bn*fidx.shape[0]],dtype=T);
    for i in range(bn*fidx.shape[0]):
        nn = i // fidx.shape[0];
        ni = i % fidx.shape[0];
        face[i] = (3,fidx[ni,0]+nn*8,fidx[ni,1]+nn*8,fidx[ni,2]+nn*8);
    pts = np.concatenate(box,axis=0);
    if colors is None:
        write_ply(path,points=pd.DataFrame(pts.astype(np.float32)),faces=pd.DataFrame(face));
    else:
        colors = np.concatenate(colors,axis=0);
        pointsc = pd.concat([pd.DataFrame(pts.astype(np.float32)),pd.DataFrame(colors)],axis=1,ignore_index=True);
        write_ply(path,points=pointsc,faces=pd.DataFrame(face),color=True);
    

def writegt(path,boxgt):
    writebox(os.path.join(path,'_002_000_gt.ply'),boxgt);
    
def writeout(path,box,color,msk):
    for i in range(1,len(box)+1):
        bout = box[0:i];
        cout = color[0:i];
        writebox(os.path.join(path,'_002_%03d_out.ply'%i),bout,cout);
        im = Image.fromarray((msk[i-1]*255).astype(np.uint8),mode='L');
        im.save(os.path.join(path,'_001_%03d_msk.png'%i));

def run(**kwargs):
    global iternum;
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
        m = importlib.import_module('net.'+opt['touch_net']);
        touchnet = m.Net(**opt);
        #
        m = importlib.import_module('net.'+opt['box_net']);
        boxnet = m.Net(**opt);
        #
        m = importlib.import_module('net.'+opt['touchpt_net']);
        touchptnet = m.Net(**opt);
        #
        if torch.cuda.is_available():
            touchnet = touchnet.cuda();
            touchnet.eval();
            boxnet = boxnet.cuda();
            boxnet.eval();
            touchptnet = touchptnet.cuda();
            touchptnet.eval();
        #
        partial_restore(touchnet,opt['touch_model']);
        partial_restore(boxnet,opt['box_model']);
        partial_restore(touchptnet,opt['touchpt_model']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    dpath = opt['data_path'];
    im_lst = os.listdir(dpath);
    for im in im_lst:
        if im.endswith('_im.png'):
            image = np.array(Image.open(os.path.join(dpath,im))).astype(np.float32) / 255.0;
            bdata = [];
            img_lst = [];
            nm_lst = [];
            smsk_lst = [];
            tmsk_lst = [];
            for msk in im_lst:
                if msk.endswith('_msk.png') and (im.split('_')[0] in msk):
                    nm_lst.append(msk);
                    img_lst.append(image);
                    mskimg = np.array(Image.open(os.path.join(dpath,msk))).astype(np.float32) / 255.0;
                    smsk_lst.append(mskimg);
                    tmsk_lst.append(mskimg);
                    print(mskimg.shape);
            img = np.stack(img_lst,axis=0);
            smsk = np.stack(smsk_lst,axis=0);
            tmsk = np.stack(tmsk_lst,axis=0);
            bdata.append(torch.from_numpy(img).cuda());
            bdata.append(torch.from_numpy(smsk).cuda());
            bdata.append(torch.from_numpy(tmsk).cuda());
            with torch.no_grad():
                boxout = boxnet(bdata);
            bx = boxout['sb'].data.cpu().numpy();
            for i,nm in enumerate(nm_lst):
                writebox(os.path.join(dpath,nm.replace('_msk.png','_box.ply')),[bx[i,...]]);
