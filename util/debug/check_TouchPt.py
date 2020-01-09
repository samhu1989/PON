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


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);

def rot(r1,r2):
    rr1 = r1 / np.sqrt(np.sum(r1**2));
    rr2 = r2 - np.sum(r2*rr1)*rr1;
    rr2 = rr2 / np.sqrt(np.sum(rr2**2));
    r3 = np.cross(rr1,rr2);
    rot = np.stack([rr1,rr2,r3],axis=0);
    return rot;
            
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
    vt = np.dot(coordt,trot.reshape(3,3)) +  + center[np.newaxis,:];
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
    iternum = opt['nepoch']
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
        train_data = m.Data(opt,'train');
        val_data = m.Data(opt,opt['user_key']);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
        
    for i, data in enumerate(train_load,0):
        data2cuda(data);
        d = data;
        break;
    #
    img = data[0].data.cpu().numpy();
    msks = data[1].data.cpu().numpy();
    mskt = data[2].data.cpu().numpy();
    vgt = data[4].data.cpu().numpy();
    #run the code
    optim = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);
    tri = box_face;
    fidx = tri;
    
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    
    debug_path = './log/debug_touchpt';
    
    for iteri in range(opt['nepoch']):
        net.train();
        out = net(data);
        loss = config.loss(data,out);
        optim.zero_grad();
        loss['overall'].backward();
        optim.step();
        net.eval();
        with torch.no_grad():
            out = net(data);
        acc = config.accuracy(data,out);
        print('iteri:',iteri);
        for k,v in acc.items():
            print(k,':',v);
        id = data[-2];
        if not os.path.exists(debug_path):
            os.mkdir(debug_path);
        tb = out['tb'].data.cpu().numpy();
        sb = out['sb'].data.cpu().numpy();
        #
        for tagi,tag in enumerate(id):
            cpath = os.path.join(debug_path,tag);
            if not os.path.exists(cpath):
                os.mkdir(cpath);
            if iteri == 0:
                im = Image.fromarray((img[tagi,...]*255).astype(np.uint8),mode='RGB');
                im.save(os.path.join(cpath,'input.png'));
                mks = Image.fromarray((msks[tagi,...]*255).astype(np.uint8),mode='L');
                mks.save(os.path.join(cpath,'msks.png'));
                mkt = Image.fromarray((mskt[tagi,...]*255).astype(np.uint8),mode='L');
                mkt.save(os.path.join(cpath,'mskt.png'));
                ptsa,ptsb = parse(vgt[tagi,...]);
                write_ply(os.path.join(cpath,'gta.ply'),points=pd.DataFrame(ptsa),faces=pd.DataFrame(face));
                write_ply(os.path.join(cpath,'gtb.ply'),points=pd.DataFrame(ptsb),faces=pd.DataFrame(face));
            write_ply(os.path.join(cpath,'a_%d.ply'%iteri),points=pd.DataFrame(sb[tagi,...]),faces=pd.DataFrame(face));
            write_ply(os.path.join(cpath,'b_%d.ply'%iteri),points=pd.DataFrame(tb[tagi,...]),faces=pd.DataFrame(face));
            
            
        