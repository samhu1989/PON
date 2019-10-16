pts_num = 1200;
pts_num_gt = 1200;
grid_num = 2;
workers = 4;
optim = 'Adam';
lr = 1e-3;
nepoch = 1;
print_epoch =1;
mode = 'Reg';
weight_decay = 0.0;
as_text = False;

from .config import parameters,NpEncoder;
import torch;

def accuracy(data,out):
    acc = {};
    L2 = ( data[6] - out['y'] )**2;
    ax = [x for x in range(1,L2.dim())];
    acc['cd'] = torch.mean(L2,dim=ax);
    return acc;
    
def loss(data,out):
    loss = {};
    L2 = ( data[6] - out['y'] )**2;
    ax = [x for x in range(1,L2.dim())];
    loss['cd'] = torch.mean(L2,dim=ax);
    loss['overall'] = torch.mean(loss['cd']);
    return loss;
    
from datetime import datetime;
from util.data.ply import write_ply
import json;
import numpy as np;
import os;
from util.tools import repeat_face,write_pts2sphere;
import pandas as pd;
from PIL import Image;
import matplotlib as mpl
mpl.use('Agg');
from util.dataset.ToyV import*;

bestcnt = 3;
best = np.array([10000]*bestcnt,dtype=np.float32);
bestn = [""]*bestcnt;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;

def writelog(**kwargs):
    global best;
    global bestn;
    opt = kwargs['opt'];
    iepoch = kwargs['iepoch'];
    nepoch = opt['nepoch'];
    ib = kwargs['idata'];
    nb = kwargs['ndata'] // opt['batch_size'];
    out = kwargs['out'];
    net = kwargs['net'];
    d = kwargs['data'];
    meter = kwargs['meter'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']));
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['config']+'_'+opt['mode']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
        os.mkdir(opt['log_tmp']);
        with open(opt['log_tmp']+os.sep+'options.json','w') as f:
            json.dump(opt,f,cls=NpEncoder);
        nparam = 0;
        with open(opt['log_tmp']+os.sep+'net.txt','w') as logtxt:
            print(str(kwargs['net']),file=logtxt);
            for p in parameters(kwargs['net']):
                nparam += torch.numel(p);
            print('nparam:%d'%nparam,file=logtxt);
        
    with open(opt['log_tmp']+os.sep+'log.txt','a') as logtxt:
        print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']),file=logtxt);
        print(json.dumps(meter,cls=NpEncoder),file=logtxt);
        
    if not kwargs['istraining'] and ib >= nb-1:
        if meter['cd'].overall_meter.avg < best[-1]:
            fn = bestn[-1];
            if fn:
                os.remove(opt['log_tmp']+os.sep+fn);
            fn = 'net_'+str(datetime.now()).replace(' ','-').replace(':','-')+'.pth';
            best[-1] = meter['cd'].overall_meter.avg;
            bestn[-1] = fn;
            torch.save(net.state_dict(),opt['log_tmp']+os.sep+fn);
            idx = np.argsort(best);
            best = best[idx];
            bestn = [bestn[x] for x in idx.tolist()];
            bestdict = dict(zip(bestn, best.tolist()));
            print(best);
            print(bestn);
            print(bestdict);
            with open(opt['log_tmp']+os.sep+'best.json','w') as f:
                json.dump(bestdict,f);

    if opt['ply'] and not kwargs['istraining']:
        ply_path = opt['log_tmp']+os.sep+'ply';
        if not os.path.exists(ply_path):
            os.mkdir(ply_path);
        img = d[0].data.cpu().numpy();
        box2d_src = d[1].data.cpu().numpy();
        box3d_src = d[2].data.cpu().numpy();
        box2d_tgt = d[3].data.cpu().numpy();
        box3d_tgt = d[4].data.cpu().numpy();
        rs = d[5].data.cpu().numpy()
        ys = out['y'].data.cpu().numpy();
        cat = d[-1];
        for i in range(img.shape[0]):
            fig = plt.figure();
            r = rs[i,...];
            y = ys[i,...];
            y *= np.pi;
            y[1] *= 2;
            coord = np.concatenate([r.reshape(-1,1),y.reshape(-1,2)],axis=1);
            c3dir = sph2car(coord.reshape(1,-1));
            c3d = np.zeros([2,3],dtype=np.float32);
            c3d[1,:] = mv(np.mean(box3d_src[i,:,:3],axis=0,keepdims=True))[:,:3];
            c3d[0,:] = c3d[1,:] + c3dir;
            c3d = mv_inv(c3d);
            c2d = proj(mv(c3d));
            ax = fig.add_subplot(121);
            ax.imshow(img[i,...]);
            ax.set_aspect('equal');
            ax.scatter(box2d_src[i,0:4,0],box2d_src[i,0:4,1],color='b',marker='*');
            ax.scatter(box2d_src[i,4:8,0],box2d_src[i,4:8,1],color='c',marker='*');
            ax.scatter(box2d_tgt[i,0:4,0],box2d_tgt[i,0:4,1],color='k',marker='x');
            ax.scatter(box2d_tgt[i,4:8,0],box2d_tgt[i,4:8,1],color='r',marker='x');
            ax.scatter(c2d[0,0],c2d[0,1],color='b',marker='o');
            ax.scatter(c2d[1,0],c2d[1,1],color='r',marker='o');
            ax = fig.add_subplot(122,projection='3d');
            #ax.set_aspect('equal');
            ax.scatter(box3d_src[i,0:4,0],box3d_src[i,0:4,1],box3d_src[i,0:4,2],color='b',marker='*');
            ax.scatter(box3d_src[i,4:8,0],box3d_src[i,4:8,1],box3d_src[i,4:8,2],color='c',marker='*');
            ax.scatter(box3d_tgt[i,0:4,0],box3d_tgt[i,0:4,1],box3d_tgt[i,0:4,2],color='k',marker='x');
            ax.scatter(box3d_tgt[i,4:8,0],box3d_tgt[i,4:8,1],box3d_tgt[i,4:8,2],color='r',marker='x');
            ax.plot(c3d[:,0],c3d[:,1],c3d[:,2]);
            cs = np.mean(box3d_src[0,:,:3],axis=0);
            ax.scatter(cs[0],cs[1],cs[2],color='r',marker='o');
            ct = np.mean(box3d_tgt[0,:,:3],axis=0);
            ax.scatter(ct[0],ct[1],ct[2],color='b',marker='o');
            plt.savefig(ply_path+os.sep+'_%04d_%03d_%s.png'%(ib,i,cat[i]));
            plt.close(fig);
    
