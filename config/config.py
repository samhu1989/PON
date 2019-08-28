import os;
import sys;
import torch;
sys.path.append('./ext/');
import cd.dist_chamfer as ext;
distChamfer =  ext.chamferDist();
from util.tools import genface,write_pts2sphere;
import pandas as pd;

def loss(data,out):
    loss = {};
    dist1, dist2 = distChamfer(data[1],out['y']);
    ax1 = [x for x in range(1,dist1.dim())]
    ax2 = [x for x in range(1,dist2.dim())]
    loss['cd'] = torch.mean(dist1,dim=ax1)+torch.mean(dist2,dim=ax2);
    loss['overall'] = (torch.mean(dist1)) + (torch.mean(dist2));
    return loss;

def accuracy(data,out):
    acc = {};
    dist1, dist2 = distChamfer(data[1],out['y']);
    ax1 = [x for x in range(1,dist1.dim())]
    ax2 = [x for x in range(1,dist2.dim())]
    acc['cd'] = torch.mean(dist1,dim=ax1)+torch.mean(dist2,dim=ax2);
    return acc;

def parameters(net):
    return net.parameters(); # train all parameters

from datetime import datetime;
from util.data.ply import write_ply
import json;
import numpy as np;

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj);

def writelog(**kwargs):
    opt = kwargs['opt'];
    iepoch = kwargs['iepoch'];
    nepoch = opt['nepoch'];
    ib = kwargs['idata'];
    nb = kwargs['ndata'] // opt['batch_size'];
    out = kwargs['out'];
    data = kwargs['data'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']));
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['mode']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
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
        print(json.dumps(kwargs['meter'],cls=NpEncoder),file=logtxt);
        
    if not kwargs['istraining']:
        if opt['ply']:
            ply_path = opt['log_tmp']+os.sep+'ply';
            if not os.path.exists(ply_path):
                os.mkdir(ply_path);
            x = out['grid_x'];
            x = x.data.cpu().numpy();
            y = out['y'];
            y = y.data.cpu().numpy();
            ygt = data[1];
            ygt = ygt.data.cpu().numpy();
            cat = data[-1];
            for i in range(y.shape[0]):
                fidx = genface(x[i,...],opt['grid_num']);
                T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
                face = np.zeros(shape=[fidx.shape[0]],dtype=T);
                for fi in range(fidx.shape[0]):
                    face[fi] = (3,fidx[fi,0],fidx[fi,1],fidx[fi,2]);
                write_ply(ply_path+os.sep+'_%04d_%03d_%s_gt.ply'%(ib,i,cat[i]),points = pd.DataFrame(ygt[i,...]));
                write_ply(ply_path+os.sep+'_%04d_%03d_%s_y.ply'%(ib,i,cat[i]),points = pd.DataFrame(y[i,...]),faces=pd.DataFrame(face));
                write_pts2sphere(ply_path+os.sep+'_%04d_%03d_%s_ypt.ply'%(ib,i,cat[i]),points = y[i,...]);
            