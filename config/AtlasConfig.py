#optim
lr = 1e-5;
optim = 'Adam'
weight_decay = 0.0
nepoch = 1
#net
pts_num = 2500
pts_num_gt = 10000
grid_num = 25
grid_dim = 2
mode = 'SVR'
#data
dataset = 'PON'
category = ['augment','Chair','StorageFurniture','Table'];
workers = 4


def loss(data,out):
    loss = {};
    return 

def accuracy(data,out):
    acc = {}
    return acc;

def parameters(net):
    return net.parameters(); # train all parameters

from datetime import datetime;
from ..util.data.ply import write_ply
def writelog(**kwargs):
    opt = kwargs['opt'];
    iepoch = kwargs['iepoch'];
    nepoch = opt['nepoch'];
    ib = kwargs['idata'] // opt['batch_size'];
    nb = kwargs['ndata'] // opt['batch_size'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb));
    if not
        
    else:
        