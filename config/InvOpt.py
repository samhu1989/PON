#optim
lr = 1e-3;
optim = 'Adam'
weight_decay = 0.0
nepoch = 5
#net
pts_num = 1000
pts_num_gt = 1000
grid_num = 1
grid_dim = 3
mode = 'InvOPT'
#data
dataset = 'PGen'
workers = 4;
from .config import accuracy,parameters,writelog;
as_text = False;
import sys;
sys.path.append('./ext/');
import cd.dist_chamfer as ext;
import torch;
distChamfer =  ext.chamferDist();

def loss(data,out):
    loss = {};
    dist1, dist2 = distChamfer(out['y'],data[1]);
    ax1 = [x for x in range(1,dist1.dim())]
    ax2 = [x for x in range(1,dist2.dim())]
    loss['cd'] = torch.mean(dist1,dim=ax1)+torch.mean(dist2,dim=ax2);
    dif = out['inv_x'] - out['grid_x'];
    ax = [x for x in range(1,dif.dim())]
    loss['inv'] = torch.mean(dif**2,dim=ax);
    loss['overall'] = torch.mean(loss['cd']) + 0.25*torch.mean(loss['inv']);
    return loss;