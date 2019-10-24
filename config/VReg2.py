pts_num = 1200;
pts_num_gt = 1200;
grid_num = 2;
workers = 6;
optim = 'Adam';
lr = 1e-3;
nepoch = 1;
print_epoch =1;
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
    L2 = ( data[6] - out['y'] )**2;
    ax = [x for x in range(1,L2.dim())];
    acc['L2'] = torch.mean(L2,dim=ax);
    vec = ( a2v(data[6]) - a2v(out['y']) )**2;
    acc['vec'] = torch.mean(vec,dim=ax);
    return acc;
    
def loss(data,out):
    loss = {};
    vec = ( a2v(data[6]) - a2v(out['y']) )**2;
    ax = [x for x in range(1,vec.dim())];
    loss['vec'] = torch.mean(vec,dim=ax);
    loss['overall'] = torch.mean(loss['vec']);
    return loss;
    
from .VReg1 import writelog;