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
    sr2 = vec[6:9];
    srot = rot(sr1,sr2);
    vs = np.dot(coords,srot.reshape(3,3));
    ts = vec[9:12];
    coordt = ts[np.newaxis,:]*coord;
    center = vec[12:15];
    tr1 = vec[15:18];
    tr2 = vec[18:21]
    trot = rot(tr1,tr2);
    vt = np.dot(coordt,trot.reshape(3,3)) + center[np.newaxis,:];
    return  vs,vt;

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
    print(opt['mode']);
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
    dpath = os.path.join(opt['data_path'],'test')
    paths = os.listdir(dpath);
    opath = './log/join'
    if not os.path.exists(opath):
        os.makedirs(opath);
    for cat in paths:
        cpath = os.path.join(opt['data_path'],'test',cat);
        copath = os.path.join(opath,cat);
        if not os.path.exists(copath):
            os.mkdir(copath);
        slst = os.listdir(cpath);
        for f in slst:
            id = os.path.basename(f).split('.')[-2];
            h5f = h5py.File(os.path.join(cpath,f),'r');
            img = np.array(h5f['img']);
            msk = np.array(h5f['msk']);
            smsk = np.array(h5f['smsk']);
            box = np.array(h5f['box']);
            num = box.shape[0];
            bdata = [];
            img_lst = [];
            smsk_lst = [];
            tmsk_lst = [];
            #for each part
            rate = opt['user_rate'];
            for i in range(num):
                msk_rate = ( np.sum(msk[i,...]) / np.sum(smsk[i,...]) );
                if msk_rate > rate:
                    img_lst.append(img);
                    smsk_lst.append(msk[i,...]);
                    tmsk_lst.append(msk[i,...]);
            #
            img = np.stack(img_lst,axis=0);
            smsk = np.stack(smsk_lst,axis=0);
            tmsk = np.stack(tmsk_lst,axis=0);
            bdata.append(torch.from_numpy(img).cuda());
            bdata.append(torch.from_numpy(smsk).cuda());
            bdata.append(torch.from_numpy(tmsk).cuda());
            with torch.no_grad():
                boxout = boxnet(bdata);
            size = np.prod(boxout['ss'].data.cpu().numpy(),axis=1);
            idx = np.argsort(-size);
            for ci in range(idx.size):
                tdata = [];
                tptdata = [];
                for cj in range(ci+1,idx.size):
                    tdata.append(  bdata[0][0,...].unsqueeze(0) );
                    tdata.append( (bdata[1][ci,...]).unsqueeze(0) );
                    tdata.append( (bdata[1][cj,...]).unsqueeze(0) );
                    with torch.no_grad():
                        touchout = touchnet(tdata);
                    if touchout['y'].data.cpu().numpy()[0][0] > 0.5:
                        tptdata.append(tdata[0]);
                        tptdata.append(tdata[1]);
                        tptdata.append(tdata[2]);
                        tptdata.append(None);
                        vec = np.zeros([1,21],dtype=np.float32);
                        vec[0,:3] = boxout['ss'].data.cpu().numpy()[ci,...];
                        vec[0,3:6] = boxout['sr1'].data.cpu().numpy()[ci,...];
                        vec[0,6:9] = boxout['sr2'].data.cpu().numpy()[ci,...];
                        vec[0,9:12] = boxout['ss'].data.cpu().numpy()[cj,...];
                        vec[0,15:18] = boxout['sr1'].data.cpu().numpy()[cj,...];
                        vec[0,18:21] = boxout['sr2'].data.cpu().numpy()[cj,...];
                        tptdata.append(torch.from_numpy(vec).cuda());
                        with torch.no_grad():
                            tptout = touchptnet(tptdata);
                        print(tptout['t'].data.cpu().numpy());
                        
            exit();
        
    '''
    print('bs:',opt['batch_size']);
        
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
           
    if opt['model']!='':
        outdir = os.path.dirname(opt['model'])+os.sep+'view';
        if not os.path.exists(outdir):
            os.mkdir(outdir);
            
    #run the code
    tri = box_face;
    fidx = tri;
    
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[2*fidx.shape[0]],dtype=T);
    for i in range(2*fidx.shape[0]):
        if i < fidx.shape[0]:
            face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
        else:
            face[i] = (3,fidx[i-fidx.shape[0],0]+8,fidx[i-fidx.shape[0],1]+8,fidx[i-fidx.shape[0],2]+8);
    '''