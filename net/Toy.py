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
import pdb
import torch.nn.functional as F;
import resnet;
import math;
from .AtlasNet import PointGenCon

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__();
        self.pts_num = kwargs['pts_num'];
        self.grid_num = kwargs['grid_num'];
        self.grid_dim = kwargs['grid_dim'];
        self.gen = PointGenCon(bottleneck_size=self.grid_dim+1024,bn=False);
        self.conv1 = torch.nn.Conv1d(3,3,1);
        self._init_layers();
        
    def forward(self,input):
        f = input[0];
        grid = self.rand_grid(f);
        expf = f.expand(f.size(0),f.size(1),grid.size(2)).contiguous();
        x = torch.cat((grid,expf),1).contiguous();
        y = self.gen(x);
        yout = y.transpose(2,1).contiguous();
        return {'y':yout};
        
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