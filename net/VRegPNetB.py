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
        self.pnet = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
            );
        
        self.mlp = nn.Sequential(
            *block(1024,256,False),
            nn.Linear(256,3),
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
        f = torch.cat([s2d.transpose(1,2).contiguous(),t2d.transpose(1,2).contiguous()],dim=2);
        x = self.pnet(f);
        x,_ = torch.max(x,2)
        x = x.view(x.size(0),-1);
        y = self.mlp(x);
        out = {'vec':y};
        out['y'] = self.v2a(y);
        return out;
        
    def v2a(self,vec):
        r = torch.sqrt(torch.sum((vec[:,:3].contiguous())**2,dim=1));
        theta = torch.acos(vec[:,2].contiguous()/r) / np.pi;
        phi = ( torch.atan2(vec[:,1].contiguous(),vec[:,0].contiguous()) + np.pi )/2.0/np.pi;
        rt = torch.stack([theta.view(-1),phi.view(-1)],dim=1);
        return rt;
    
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
