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
    layers.append(nn.ReLU(inplace=True))
    return layers;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__();
        self.model = nn.Sequential(
            *block(625,1024,False),
            *block(1024,256,False),
            *block(256,2,False)
        );
        self._init_layers();

    def forward(self,input):
        s2d = input[1];
        t2d = input[3];
        s = np.ones([1,1,3],dtype=np.float32);
        s[:,:,:2] *= 112.0;
        s = torch.from_numpy(s);
        s2d = s2d / s.type(s2d.type());
        t2d = t2d / s.type(t2d.type());
        xs = s2d.view(s2d.size(0),-1,1);
        xt = t2d.view(t2d.size(0),1,-1);
        xs = torch.cat([xs,torch.ones(s2d.size(0),1,1).type(s2d.type())],dim=1);
        xt = torch.cat([xt,torch.ones(t2d.size(0),1,1).type(t2d.type())],dim=2);
        f = torch.bmm(xs,xt);
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
