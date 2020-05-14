#optim
lr = 1e-5;
optim = 'Adam'
weight_decay = 0.0
nepoch = 4
#net
pts_num = 2500
pts_num_gt = 10000
grid_num = 1
grid_dim = 3
mode = 'SVR'
#data
dataset = 'atlas'
category = None;
workers = 1

from .config import accuracy,loss,parameters,writelog;
        