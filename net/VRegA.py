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

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__();
        self.model = nn.Sequential(
            *block(32,256,False),
            *block(256,128,False),
            *block(128,64,False),
            nn.Linear(64,2),
            #nn.Hardtanh(min_val=0.0,max_val=1.0,inplace=True)
            nn.Sigmoid()
        );
        self._init_layers();

    def forward(self,input):
        s2d = input[1];
        t2d = input[3];
        xs = s2d.view(s2d.size(0),-1) / 112.0;
        xt = t2d.view(t2d.size(0),-1) / 112.0;
        f = torch.cat([xs,xt],dim=1);
        f = f.view(f.size(0),-1).contiguous();
        y = self.model(f);
        out = {'y':y};
        return out;
    
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,np.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();
