import os;
import sys;
import torch;
import pandas as pd;
from util.tools import write_tfb_loss;
from datetime import datetime;
import json;
import numpy as np;
from .config import NpEncoder;

beta = 1;
input_size = 24;
latent_size = 1024;
z_size = 8;
part_idx = [i for i in range(0,input_size//2)];
part_idx.extend( [i for i in range(input_size//2+3,input_size-6)] );
workers = 4;
lr = 1e-3;
weight_decay = 0.0;
nepoch = 1000;
category = ['Chair','Table','StorageFurniture','Bed','Display'];

def loss(data,out):
    x = data[0];
    recon_x = out['rx'];
    mu = out['mu'];
    logvar = out['logvar'];
    loss = {};
    loss['kl'] =  ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / x.size(0);
    loss['recon'] = torch.sum( ( recon_x - x )**2 ) / x.size(0) ;
    loss['betakl'] = beta*loss['kl'];
    loss['overall'] = loss['betakl'] + loss['recon'];
    return loss;
    
def parameters(net):
    return net.parameters(); # train all parameters
    
def writelog(**kwargs):
    opt = kwargs['opt'];
    iepoch = kwargs['iepoch'];
    nepoch = opt['nepoch'];
    ib = kwargs['idata'];
    nb = kwargs['ndata'] // opt['batch_size'];
    out = kwargs['out'];
    net = kwargs['net'];
    data = kwargs['data'];
    loss = kwargs['loss'];
    optim = kwargs['optim'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']));
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['config']+'_'+opt['dataset']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
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
        info = "";
        for k,v in loss.items():
            info += k + ":" + str(v.data.cpu().numpy()) + ","
        print(info,file=logtxt);
        print(data[1],file=logtxt);
        print(info);
        

    tfb_dir = os.path.join(opt['log_tmp'],'tfb');
    write_tfb_loss(tfb_dir,loss,ib+nb*iepoch,nb,optim);
        