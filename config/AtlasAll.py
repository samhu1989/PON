#optim
lr = 1e-5;
optim = 'Adam'
weight_decay = 0.0
nepoch = 2048
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

from .config import accuracy,loss,parameters,writelog;
        