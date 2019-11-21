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
    
def fetch(x,coord):
    w = x.size(2);
    h = x.size(3);
    f = x.view(x.size(0),x.size(1),-1);
    index = torch.floor(coord[:,:,0].contiguous()*(w//2))*h + torch.floor(coord[:,:,1].contiguous()*(h//2));
    index = index.unsqueeze(1).expand(-1,x.size(1),-1).contiguous();
    return torch.gather(f,2,index.long()).contiguous();

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__();
        self.resnet = resnet.resnet18(pretrained=False,input_channel=5,num_classes=625);
        self.model = nn.Sequential(
            *block(625,1024,False),
            *block(1024,256,False),
            nn.Linear(256,3)
        );
        self._init_layers();

    def forward(self,input):
        img = input[0];
        x = img[:,:,:,:3].contiguous();
        x = x.permute(0,3,1,2).contiguous();
        #
        ms = input[7].unsqueeze(1);
        mt = input[8].unsqueeze(1);
        #
        xs = torch.cat([x,ms,mt],dim=1).contiguous();
        #
        f = self.resnet(xs);
        #
        f = f.view(f.size(0),-1).contiguous();
        #
        y = self.model(f);
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