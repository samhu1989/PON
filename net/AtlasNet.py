from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(__file__))
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim;
from torch.nn.parameter import Parameter
import torch.utils.data
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F;
import resnet;
import math;

def resize(*input):
    y = input[0];
    size = input[1:];
    if isinstance(size[0],torch.Size):
        size = size[0];
    if isinstance(y,torch.FloatTensor) or isinstance(y,torch.cuda.FloatTensor):
        y = y.resize_(size);
    else:
        y = y.resize(*size);
    return y;
    
def interp(x,Li,Lw):
    if isinstance(x,Variable) and not isinstance(Li,Variable):
        Li = Variable(Li);
        Lw = Variable(Lw);
    if x.is_cuda:
        Li = Li.cuda();
        Lw = Lw.cuda();
    Lx = x;
    Lx = Lx.index_select(dim=2,index=Li).contiguous();
    Lw = Lw / Lw.sum(dim=-1,keepdim=True);
    Lw = resize(Lw,Lw.size()[0],1,Lw.size()[1],Lw.size()[-1]);
    Lx = resize(Lx,Lx.size()[0],Lx.size()[1],Lx.size()[2]/Lw.size()[-1],Lw.size()[-1]);
    Lx = resize((Lw*Lx).sum(dim=-1).contiguous(),x.size());
    return Lx;

#atlasnet
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500,odim=3,bn=True):
        self.bottleneck_size = bottleneck_size;
        super(PointGenCon, self).__init__();
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size,1);
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1);
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2,self.bottleneck_size//4,1);
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4,odim,1);
        self.bn = bn;
        self.th = nn.Tanh();
        if self.bn:
            self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size);
            self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2);
            self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4);
        
    def forward(self, x):
        # print(x.size());
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)));
            x = F.relu(self.bn2(self.conv2(x)));
            x = F.relu(self.bn3(self.conv3(x)));
            x = self.th(self.conv4(x))
        else:
            x = F.relu(self.conv1(x));
            x = F.relu(self.conv2(x));
            x = F.relu(self.conv3(x));
            x = self.th(self.conv4(x))
        return x;
    
class OptEncoder(nn.Module):
    def __init__(self,bottleneck_size=1024):
        self.bottleneck_size = bottleneck_size;
        super(OptEncoder,self).__init__();
        self.f = Parameter(torch.zeros(1,self.bottleneck_size));
        self.f.data.normal_(0,math.sqrt(2./float(self.bottleneck_size)));
        
    def forward(self,x):
        return self.f;
    
class OptDecoder(nn.Module):
    def __init__(self,bottleneck_size=1024):
        self.bottleneck_size = bottleneck_size;
        super(OptDecoder,self).__init__();
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size,self.bottleneck_size//2,1);
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size,3,1);
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size//2);
        self.th = nn.Tanh();
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)));
        return self.th(self.conv2(x));
        
class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__();
        self.pretrained_encoder = False;
        self.pts_num = kwargs['pts_num']
        self.bottleneck_size = 1024
        self.grid_num = kwargs['grid_num']
        self.grid_dim = kwargs['grid_dim']
        self.mode = kwargs['mode']
        self.bn = True
        self.inv_y = None;
        if self.mode == 'SVR' or 'InvSVR':
            self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder,num_classes=1024);
        elif self.mode == 'AE' or 'InvAE':
            self.encoder = nn.Sequential(
                PointNetfeat(self.pts_num, global_feat=True, trans = False),
                nn.Linear(1024, self.bottleneck_size),
                nn.BatchNorm1d(self.bottleneck_size),
                nn.ReLU()
                );
        elif self.mode == 'OPT' or 'InvOPT':
            self.bn = False;
            self.encoder = OptEncoder(self.bottleneck_size);
        else:
            assert False,'unkown mode of InvAtlasNet'
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=self.grid_dim+self.bottleneck_size,bn=self.bn) for i in range(0,self.grid_num)]);
        if self.mode.startswith('Inv'):
            self.inv_decoder = nn.ModuleList([PointGenCon(bottleneck_size=3+self.bottleneck_size,odim=self.grid_dim,bn=self.bn)]);
        self._init_layers();

    def forward(self,input):
        x = input[0];
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous();
        f = self.encoder(x);
        grid = self.rand_grid(f);
        expf = f.unsqueeze(2).expand(f.size(0),f.size(1),grid.size(2)).contiguous();
        outs = [];
        for i in range(0,self.grid_num):
            y = torch.cat((grid,expf),1).contiguous();
            y = self.decoder[i](y);
            outs.append(y);
        y = torch.cat(outs,2).contiguous()
        yout = y.transpose(2,1).contiguous();
        if grid.size(2) != y.size(2):
            expf = f.unsqueeze(2).expand(f.size(0),f.size(1),y.size(2)).contiguous();
        out = {}
        out['y'] = yout
        if self.mode.startswith('Inv'):
            inv_y = torch.cat((y,expf),1).contiguous();
            inv_y = self.inv_decoder[0](inv_y);
            inv_y = inv_y.transpose(2,1).contiguous()
            out['inv_x'] = inv_y
        if self.grid_num > 1:
            out['grid_x'] = torch.cat([grid  for i in range(0,self.grid_num)],2).contiguous().transpose(2,1).contiguous(); 
        else:
            out['grid_x'] = grid.transpose(2,1).contiguous();
        return out;
    
    def rand_grid(self,x):
        rand_grid = torch.FloatTensor(x.size(0),self.grid_dim,self.pts_num//self.grid_num);
        if self.grid_dim == 3:
            rand_grid.normal_(0.0,1.0);
            rand_grid += 1e-9;
            rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True);
        else:
            rand_grid.uniform_(0.0,1.0);
        if isinstance(x,Variable):
            rand_grid = Variable(rand_grid);
        if x.is_cuda:
            rand_grid = rand_grid.cuda();
        return rand_grid;
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,math.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();