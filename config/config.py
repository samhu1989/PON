import os;
import sys;
import torch;
sys.path.append('./ext/');
from util.tools import genface,write_pts2sphere;
import pandas as pd;
as_text = False;

def loss(data,out):
    import cd.dist_chamfer as ext;
    distChamfer =  ext.chamferDist();
    loss = {};
    dist1, dist2 = distChamfer(data[1],out['y']);
    ax1 = [x for x in range(1,dist1.dim())]
    ax2 = [x for x in range(1,dist2.dim())]
    loss['cd'] = torch.mean(dist1,dim=ax1)+torch.mean(dist2,dim=ax2);
    loss['overall'] = (torch.mean(dist1)) + (torch.mean(dist2));
    return loss;

def accuracy(data,out):
    import cd.dist_chamfer as ext;
    distChamfer =  ext.chamferDist();
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
            
from PIL import Image;

bestcnt = 3;
best = np.array([10000]*bestcnt,dtype=np.float32);
bestn = [""]*bestcnt;

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
    data = kwargs['data'];
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
        x = out['grid_x'];
        x = x.data.cpu().numpy();
        y = out['y'];
        y = y.data.cpu().numpy();
        ygt = data[1];
        ygt = ygt.data.cpu().numpy();
        cat = data[-1];
        im = data[0];
        im = im.data.cpu().numpy();
        for i in range(y.shape[0]):
            fidx = genface(x[i,...],opt['grid_num']);
            T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
            face = np.zeros(shape=[fidx.shape[0]],dtype=T);
            for fi in range(fidx.shape[0]):
                face[fi] = (3,fidx[fi,0],fidx[fi,1],fidx[fi,2]);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_gt.ply'%(ib,i,cat[i]),points = pd.DataFrame(ygt[i,...]));
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_y.ply'%(ib,i,cat[i]),points = pd.DataFrame(y[i,...]),faces=pd.DataFrame(face),as_text=opt['as_text']);
            write_pts2sphere(ply_path+os.sep+'_%04d_%03d_%s_ypt.ply'%(ib,i,cat[i]),points = y[i,...]);
            img = im[i,...];
            img = img.transpose((1,2,0));
            img = Image.fromarray(np.uint8(255.0*img));
            img.save(ply_path+os.sep+'_%04d_%03d_%s_input.png'%(ib,i,cat[i]));
                
                
            