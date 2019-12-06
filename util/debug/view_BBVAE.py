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
from ..data.gen_toybox import box_face;
from net.g.box import Box;
from util.dataset.ToyV import recon,mv,proj;
from util.tools import partial_restore;
from ..data.obb import OBB;
import os;

def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);
            
def normalize(v):
    norm = np.linalg.norm(v);
    if norm == 0: 
       return v
    return v / norm;
    
def proj(v2,v1):
    return np.dot(v2,v1)*v1;
    
def deal(x):
    v1 = x[6:9];
    x[6:9] = normalize(v1);
    v1 = x[6:9];
    v2 = x[9:12];
    v2 = v2 - proj(v2,v1);
    x[9:12] = normalize(v2);
    #
    v1 = x[18:21];
    x[18:21] = normalize(v1);
    v1 = x[18:21];
    v2 = x[21:24];
    v2 = v2 - proj(v2,v1);
    x[21:24] = normalize(v2);
    
    
def parse(x):
    deal(x);
    #=================
    obba = OBB();
    ca = x[:3];
    ea = x[3:6];
    ra = x[6:12];
    ra3 = np.cross(x[6:9],x[9:12]); 
    r = np.zeros((3,3),dtype=np.float32);
    r[0,:] = x[6:9];
    r[1,:] = x[9:12];
    r[2,:] = ra3;
    obba.rotation = r;
    centroid = np.dot(ca,np.linalg.inv(r));
    obba.min = centroid - ea;
    obba.max = centroid + ea;
    ptsa = np.stack(obba.points,axis=0).astype(np.float32);
    #=================
    obbb = OBB();
    cb = x[12:15];
    eb = x[15:18];
    rb = x[18:24];
    rb3 = np.cross(x[18:21],x[21:24]); 
    r = np.zeros((3,3),dtype=np.float32);
    r[0,:] = x[18:21];
    r[1,:] = x[21:24];
    r[2,:] = rb3;
    obbb.rotation = r;
    centroid = np.dot(cb,np.linalg.inv(r));
    obbb.min = centroid - eb;
    obbb.max = centroid + eb;
    ptsb = np.stack(obbb.points,axis=0).astype(np.float32);
    return ptsa,ptsb;


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
        train_data = m.Data(opt,True);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
        
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
        
    bi = int(opt['user_key']);    
    if opt['model']!='':
        outdir = os.path.dirname(opt['model'])+os.sep+'view_%d'%bi;
        if not os.path.exists(outdir):
            os.mkdir(outdir); 
    #
    input_size = opt['input_size'];
    idx = opt['part_idx'];
    tri = box_face;
    
    for i, data in enumerate(train_load,0):
        print(i,'/',len(train_data)//opt['batch_size']);
        data2cuda(data);
        net.eval();
        zs = np.linspace(-3,3,20).astype(np.float32);
        xx = data[0][bi,idx].contiguous().view(1,-1);
        ox = data[0][bi,:].contiguous().view(1,-1);
        cat = data[1][bi];
        print(cat);
        #==================================
        ptsa, ptsb = parse(ox.data.cpu().numpy()[0,:]);
        #==================================
        fig = plt.figure(figsize=(9.6,4.8));
        ax = fig.add_subplot(1,2,1,projection='3d');
        ax.view_init(elev=20, azim=90)
        ax.set_aspect('equal', adjustable='box');
        ax.set_xlim([-1,1]);
        ax.set_ylim([-1,1]);
        ax.set_zlim([-1,1]);
        #
        ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
        ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
        
        ax = fig.add_subplot(1,2,2,projection='3d');
        ax.set_aspect('equal', adjustable='box');
        ax.set_xlim([-1,1]);
        ax.set_ylim([-1,1]);
        ax.set_zlim([-1,1]);
        #
        ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
        ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
        plt.savefig(os.path.join(outdir,'_%d_%s_input.png'%(i,cat)));
        plt.close(fig);
        for zi in range(opt['z_size']):
            for i in range(20):
                fig = plt.figure(figsize=(9.6,4.8));
                z = np.zeros([1,opt['z_size']],dtype=np.float32);
                z[0,zi] = zs[i];
                z = torch.from_numpy(z).cuda();
                with torch.no_grad(): 
                    r = net.decode(xx,z);
                x = r.data.cpu().numpy()[0,:];
                ptsa, ptsb = parse(x);
                #==================
                ax = fig.add_subplot(1,2,1,projection='3d');
                ax.view_init(elev=20, azim=90)
                ax.set_aspect('equal', adjustable='box');
                ax.set_xlim([-1,1]);
                ax.set_ylim([-1,1]);
                ax.set_zlim([-1,1]);
                #
                ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
                ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
                #
                ax = fig.add_subplot(1,2,2,projection='3d');
                ax.set_aspect('equal', adjustable='box');
                ax.set_xlim([-1,1]);
                ax.set_ylim([-1,1]);
                ax.set_zlim([-1,1]);
                #
                ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
                ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
                plt.savefig(os.path.join(outdir,'_%d_bbvae_%d_%f.png'%(i,zi,zs[i])));
                plt.close(fig);
        
