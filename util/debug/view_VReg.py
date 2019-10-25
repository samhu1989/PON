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
import os;


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);

def init():
    return pv;

def animate(i,config,net,optim,data):
    print(i,'/',iternum);
    net.train();
    out = net(data);
    loss = config.loss(data,out);
    optim.zero_grad();
    loss['overall'].backward();
    optim.step();
    print(loss['overall'].data.cpu().numpy());
    net.eval();
    with torch.no_grad():
        out = net(data);
    box2d_src = data[1].data.cpu().numpy();
    box3d_src = data[2].data.cpu().numpy();
    box2d_tgt = data[3].data.cpu().numpy();
    box3d_tgt = data[4].data.cpu().numpy();
    r = data[5].data.cpu().numpy();
    gts = data[6].data.cpu().numpy();
    y = out['y'].data.cpu().numpy();
    num = box3d_src.shape[0];
    col = 8;
    row = num // col;
    for ri in range(row):
        for cj in range(col):
            ni = ri*col+cj;
            ymap = y[ni,...];
            ymap *= np.pi;
            ymap[1] *= 2;
            c3d = recon(box3d_src[ni,...],r[ni,...],ymap);
            pv[ni].set_data(c3d[:,0],c3d[:,1]);
            pv[ni].set_3d_properties(c3d[:,2]);
    if i == iternum-1:
        exit();
    return pv;

def run(**kwargs):
    print('a')
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
        val_data = m.Data(opt,False);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
        
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
    if 'train' in opt['user_key']:
        load = train_load;
    else:
        load = val_load;
    outdir = os.path.dirname(opt['model'])+os.sep+'view_'+opt['user_key'];
    if not os.path.exists(outdir):
        os.mkdir(outdir);
    for i, data in enumerate(load,0):
        data2cuda(data);
        net.eval();
        with torch.no_grad():
            out = net(data);
        img = data[0].data.cpu().numpy();
        box2d_src = data[1].data.cpu().numpy();
        box3d_src = data[2].data.cpu().numpy();
        box2d_tgt = data[3].data.cpu().numpy();
        box3d_tgt = data[4].data.cpu().numpy();
        r = data[5].data.cpu().numpy();
        gts = data[6].data.cpu().numpy();
        y = out['y'].data.cpu().numpy();

        tri = box_face;
        num = box3d_src.shape[0];
        col = 4;
        row = num // col;
        for ri in range(row):
            for cj in range(col):
                ni = ri*col+cj;
                fig = plt.figure(figsize=(48,16));
                #
                ax = fig.add_subplot(132,projection='3d');
                ax.view_init(elev=20, azim=0)
                ax.plot_trisurf(box3d_tgt[ni,...,0],box3d_tgt[ni,...,1],tri,box3d_tgt[ni,...,2],color=(0,0,1,0.1));
                ax.plot_trisurf(box3d_src[ni,...,0],box3d_src[ni,...,1],tri,box3d_src[ni,...,2],color=(0,1,0,0.1));
                ygt = gts[ni,...];
                ygt *= np.pi;
                ygt[1] *= 2;
                c3d1 = recon(box3d_src[ni,...],r[ni,...],ygt);
                ax.plot(c3d1[:,0],c3d1[:,1],c3d1[:,2],color='k');
                ymap = y[ni,...];
                ymap *= np.pi;
                ymap[1] *= 2;
                c3d2 = recon(box3d_src[ni,...],r[ni,...],ymap);
                ax.plot(c3d2[:,0],c3d2[:,1],c3d2[:,2],color='r');
                ax.scatter(box3d_src[ni,0:4,0],box3d_src[ni,0:4,1],box3d_src[ni,0:4,2],color='b',marker='*');
                ax.scatter(box3d_src[ni,4:8,0],box3d_src[ni,4:8,1],box3d_src[ni,4:8,2],color='c',marker='*');
                ax.scatter(box3d_tgt[ni,0:4,0],box3d_tgt[ni,0:4,1],box3d_tgt[ni,0:4,2],color='k',marker='x');
                ax.scatter(box3d_tgt[ni,4:8,0],box3d_tgt[ni,4:8,1],box3d_tgt[ni,4:8,2],color='r',marker='x');
                ax.set_aspect('equal', adjustable='box');
                #
                ax = fig.add_subplot(133,projection='3d');
                ax.view_init(elev=20, azim=90)
                ax.plot_trisurf(box3d_tgt[ni,...,0],box3d_tgt[ni,...,1],tri,box3d_tgt[ni,...,2],color=(0,0,1,0.1));
                ax.plot_trisurf(box3d_src[ni,...,0],box3d_src[ni,...,1],tri,box3d_src[ni,...,2],color=(0,1,0,0.1));
                ax.plot(c3d1[:,0],c3d1[:,1],c3d1[:,2],color='k');
                ax.plot(c3d2[:,0],c3d2[:,1],c3d2[:,2],color='r');
                ax.scatter(box3d_src[ni,0:4,0],box3d_src[ni,0:4,1],box3d_src[ni,0:4,2],color='b',marker='*');
                ax.scatter(box3d_src[ni,4:8,0],box3d_src[ni,4:8,1],box3d_src[ni,4:8,2],color='c',marker='*');
                ax.scatter(box3d_tgt[ni,0:4,0],box3d_tgt[ni,0:4,1],box3d_tgt[ni,0:4,2],color='k',marker='x');
                ax.scatter(box3d_tgt[ni,4:8,0],box3d_tgt[ni,4:8,1],box3d_tgt[ni,4:8,2],color='r',marker='x');
                ax.set_aspect('equal', adjustable='box');
                #
                ax = fig.add_subplot(131);
                ax.set_aspect('equal', adjustable='box');
                ax.imshow(img[ni,...]);
                ax.scatter(box2d_src[ni,0:4,0],box2d_src[ni,0:4,1],color='b',marker='*');
                ax.scatter(box2d_src[ni,4:8,0],box2d_src[ni,4:8,1],color='c',marker='*');
                ax.scatter(box2d_tgt[ni,0:4,0],box2d_tgt[ni,0:4,1],color='k',marker='x');
                ax.scatter(box2d_tgt[ni,4:8,0],box2d_tgt[ni,4:8,1],color='r',marker='x');
                ax.set_aspect('equal', adjustable='box');
                c2d1 = proj(mv(c3d1));
                c2d2 = proj(mv(c3d2));
                ax.plot(c2d1[:,0],c2d1[:,1],color='k');
                ax.plot(c2d2[:,0],c2d2[:,1],color='r');
                plt.savefig(os.path.join(outdir,"_%04d_%04d.png"%(i,ni)));
                if opt['ply']:
                    plt.show();
                plt.close(fig);
            
    #run the code
    '''
    Writer = animation.writers['ffmpeg'];
    writer = Writer(fps=30, metadata=dict(artist='Siyu'));
    anim.save('./log/debug_VReg.mp4',writer=writer);
    '''