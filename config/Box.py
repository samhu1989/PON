import os;
import sys;
import torch;
import pandas as pd;
from util.tools import write_tfb;
from datetime import datetime;
import json;
import numpy as np;
from .config import NpEncoder;
import torch.nn as nn;

workers = 4;
lr = 1e-3;
weight_decay = 0.0;
nepoch = 1000;
print_epoch = 1;

def rot(x_raw,y_raw):
        x = F.normalize(x_raw,dim=1,p=2);
        z = torch.cross(x,y_raw);
        z = F.normalize(z,dim=1,p=2);
        y = torch.cross(z,x);
        rot = torch.stack([x,y,z],dim=1);
        rot = rot.view(-1,3,3);
    return rot;

def sr2box(size,r1,r2):
    const = np.array([[[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]]],dtype=np.float32);
    const = torch.from_numpy(const);
    const = const.type(size.type());
    const = const.requires_grad = True;
    rot = self.rot(r1,r2);
    box = const*( size.unsqueeze(1).contiguous() );
    box = torch.matmul(box,rot);
    return box;

def loss(data,out):
    vgt = data[4];
    #
    ss_gt = vgt[:,0:3];
    sr1_gt = vgt[:,3:6];
    sr2_gt = vgt[:,6:9];
    sb_gt = sr2box(ss_gt,sr1_gt,sr2_gt); 
    #
    ts_gt = vgt[:,9:12];
    tr1_gt = vgt[:,15:18];
    tr2_gt = vgt[:,18:21];
    tb_gt = sr2box(ts_gt,st1_gt,tr2_gt);
    #
    ss = out['ss'];
    sr1 = out['sr1'];
    sr2 = out['sr2'];
    sb = out['sb'];
    #
    ts = out['ts'];
    tr1 = out['tr1'];
    tr2 = out['tr2'];
    tb = out['tb'];
    #
    loss = {};
    loss['size'] = 0.5*torch.sum( ( ss - ss_gt.data )**2 + ( ts - ts_gt.data )**2, dim = 1 );
    rot = ( sr1 - sr1_gt.data )**2;
    rot += ( sr2 - sr2_gt.data )**2;
    rot += ( tr1 - tr1_gt.data )**2;
    rot += ( tr2 - tr2_gt.data )**2;
    loss['rot6'] = 0.5*torch.sum( rot, dim=1 );
    loss['box'] = 0.5*torch.sum( ( sb - sb_gt.data )**2 + ( tb - tb_gt.data )**2, dim = 1 );
    loss['overall'] = torch.mean( loss['box'] );
    return loss;
    
def accuracy(data,out):
    vgt = data[4];
    #
    ss_gt = vgt[:,0:3];
    sr1_gt = vgt[:,3:6];
    sr2_gt = vgt[:,6:9];
    sb_gt = sr2box(ss_gt,sr1_gt,sr2_gt); 
    #
    ts_gt = vgt[:,9:12];
    tr1_gt = vgt[:,15:18];
    tr2_gt = vgt[:,18:21];
    tb_gt = sr2box(ts_gt,tr1_gt,tr2_gt);
    #
    ss = out['ss'];
    sr1 = out['sr1'];
    sr2 = out['sr2'];
    sb = out['sb'];
    #
    ts = out['ts'];
    tr1 = out['tr1'];
    tr2 = out['tr2'];
    tb = out['tb'];
    #
    loss = {};
    loss['size'] = 0.5*torch.sum( ( ss - ss_gt.data )**2 + ( ts - ts_gt.data )**2, dim = 1 );
    erot = ( sr1 - sr1_gt.data )**2;
    erot += ( sr2 - sr2_gt.data )**2;
    erot += ( tr1 - tr1_gt.data )**2;
    erot += ( tr2 - tr2_gt.data )**2;
    loss['rot6'] = 0.5*torch.sum( erot, dim=1 );
    loss['box'] = 0.5*torch.sum( ( sb - sb_gt.data )**2 + ( tb - tb_gt.data )**2, dim = 1 );
    loss['overall'] = loss['box'];
    return loss;
    
def parameters(net):
    return net.parameters(); # train all parameters
    
bestcnt = 3;
best = np.array([10000.0]*bestcnt,dtype=np.float32);
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
    d = kwargs['data'];
    meter = kwargs['meter'];
    optim = kwargs['optim'];
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
    if kwargs['istraining']:
        tfb_dir = os.path.join(opt['log_tmp'],'tfb','train');
    else:
        tfb_dir = os.path.join(opt['log_tmp'],'tfb','test');
    write_tfb(tfb_dir,meter,ib+nb*iepoch,nb,optim);
    
    if not kwargs['istraining'] and ib >= nb-1:
        if meter['box'].overall_meter.avg < best[-1]:
            fn = bestn[-1];
            if fn:
                os.remove(opt['log_tmp']+os.sep+fn);
            fn = 'net_'+str(datetime.now()).replace(' ','-').replace(':','-')+'.pth';
            best[-1] = meter['box'].overall_meter.avg;
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