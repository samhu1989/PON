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
        self.resnet = resnet.resnet18(pretrained=False,num_classes=1024,fc=False);
        self.pnet = nn.Sequential(
            nn.Conv1d(1024,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,32,1),
            nn.BatchNorm1d(32)
        );
        self.down = nn.Sequential(
            nn.Conv1d(1024,32,1),
            nn.BatchNorm1d(32)
        );
        self.fout = nn.Conv1d(32,1,1);
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
        x = self.resnet.conv1(x);
        x = self.resnet.bn1(x);
        x = self.resnet.relu(x); #112x112
        #
        x0 = self.resnet.maxpool(x); #64x56x56
        # 
        x1 = self.resnet.layer1(x0); #64x56x56
        x2 = self.resnet.layer2(x1); #128x28x28
        x3 = self.resnet.layer3(x2); #256x14x14
        x4 = self.resnet.layer4(x3); #512x7x7
        #
        s2d = input[1][:,:,:2].contiguous();
        t2d = input[3][:,:,:2].contiguous();
        s = np.ones([1,1,2],dtype=np.float32);
        s[:,:,:2] *= 112.0;
        s = torch.from_numpy(s);
        #
        s2d = s2d / s.type(s2d.type());
        t2d = t2d / s.type(t2d.type());
        #
        sy0 = fetch(x0,s2d);
        ty0 = fetch(x0,t2d);
        sy1 = fetch(x1,s2d);
        ty1 = fetch(x1,t2d);
        sy2 = fetch(x2,s2d);
        ty2 = fetch(x2,t2d);
        sy3 = fetch(x3,s2d);
        ty3 = fetch(x3,t2d);
        sy4 = fetch(x4,s2d);
        ty4 = fetch(x4,t2d);
        #
        fs = torch.cat([sy0, sy1, sy2, sy3, sy4],dim=1).contiguous();
        ft = torch.cat([ty0, ty1, ty2, ty3, ty4],dim=1).contiguous();
        #
        fs = self.fout( self.down(fs) + self.pnet(fs) );
        ft = self.fout( self.down(ft) + self.pnet(ft) );
        #
        xs = s2d.transpose(1,2).contiguous();
        xt = t2d.transpose(1,2).contiguous();
        #
        xs = torch.cat([xs,fs],dim=1);
        xt = torch.cat([xt,ft],dim=1);
        #
        xs = xs.view(xs.size(0),-1,1);
        xt = xt.view(xt.size(0),1,-1);
        #
        xs = torch.cat([xs,torch.ones(xs.size(0),1,1).type(xs.type())],dim=1);
        xt = torch.cat([xt,torch.ones(xt.size(0),1,1).type(xt.type())],dim=2);
        f = torch.bmm(xs,xt);
        f = f.view(f.size(0),-1).contiguous();
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
