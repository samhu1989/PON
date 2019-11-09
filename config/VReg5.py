pts_num = 1200;
pts_num_gt = 1200;
grid_num = 2;
workers = 8;
optim = 'Adam';
lr = 1e-3;
nepoch = 1;
print_epoch = 1;
mode = 'Reg';
weight_decay = 0.0;
as_text = False;

from .config import parameters,NpEncoder;
import torch;
import numpy as np;

def a2v(vec):
    v = vec*np.pi;
    th = v[:,0].contiguous();
    phi = (v[:,1]*2.0).contiguous();
    sth = torch.sin(th);
    cth = torch.cos(th);
    sphi = torch.sin(phi - np.pi);
    cphi = torch.cos(phi - np.pi);
    x = sth*cphi;
    y = sth*sphi;
    z = cth;
    res = torch.stack([x,y,z],dim=1);
    return res; 

def accuracy(data,out):
    acc = {};
    L2 = ( a2v(data[5]) - out['vec'] )**2;
    ax = [x for x in range(1,L2.dim())];
    acc['cd'] = torch.mean(L2,dim=ax);
    return acc;
    
def loss(data,out):
    loss = {};
    L2 = ( a2v(data[5]) - out['vec'] )**2;
    ax = [x for x in range(1,L2.dim())];
    loss['cd'] = torch.mean(L2,dim=ax);
    loss['overall'] = torch.mean(loss['cd']);
    return loss;
    
from .VReg1 import writelog;