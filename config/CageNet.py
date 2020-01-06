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

def loss(data,out):
    ygt = data[3];
    vgt = data[4];
    yout = out['y'];
    vout = out['vec'];
    loss = {};
    bce = nn.BCELoss(reduction='none')
    loss['bce'] = bce(yout,ygt.data);
    loss['recon'] = torch.sum(( vgt - vout )**2,dim=1,keepdims=True);
    loss['overall'] = torch.mean(loss['bce'] + ygt*loss['recon']);
    return loss;
    
def accuracy(data,out):
    ygt = data[3];
    vgt = data[4];
    yout = out['y'];
    vout = out['vec'];
    loss = {};
    bce = nn.BCELoss(reduction='none')
    loss['bce'] = bce(yout,ygt.data);
    loss['acc'] = (( yout.data > 0.5) == ygt.data ); 
    loss['recon'] = torch.sum(( vgt - vout )**2,dim=1,keepdims=True);
    loss['overall'] = loss['bce'] + ygt*loss['recon'];
    return loss;
    
def parameters(net):
    return net.parameters(); # train all parameters
    
bestcnt = 1;
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
    d = kwargs['data'];
    meter = kwargs['meter'];
    optim = kwargs['optim'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']));
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['config']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
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
        if meter['overall'].overall_meter.avg < best[-1]:
            fn = bestn[-1];
            if fn:
                os.remove(opt['log_tmp']+os.sep+fn);
            fn = 'net_'+str(datetime.now()).replace(' ','-').replace(':','-')+'.pth';
            best[-1] = meter['overall'].overall_meter.avg;
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