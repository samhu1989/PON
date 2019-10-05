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
from .AtlasNet import Net as AtlasNet;
        
class Net(AtlasNet):
    def __init__(self,**kwargs):
        super(Net, self).__init__(**kwargs);
        if ( self.mode == 'SVR' ) or ( self.mode =='InvSVR' ):
            self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder,input_channel=5,num_classes=1024);

    def forward(self,input):
        x = input[0];
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous();
            coord = torch.from_numpy((np.mgrid[-1:1:224j,-1:1:224j]).astype(np.float32));
            coord = coord.unsqueeze(0).expand(x.size(0),coord.size(0),coord.size(1),coord.size(2));
            coord = coord.type(x.type());
            x = torch.cat((coord,x),1).contiguous();
            
        f = self.encoder(x);
        grid = self.rand_grid(f);
        expf = f.unsqueeze(2).expand(f.size(0),f.size(1),grid.size(2)).contiguous();
        outs = [];
        for i in range(0,self.grid_num):
            y = torch.cat((grid,expf),1).contiguous();
            y = self.decoder[i](y);
            outs.append(y);
        yout = torch.cat(outs,2).contiguous();
        yout = yout.transpose(2,1).contiguous();
        if grid.size(2) != y.size(2):
            expf = f.unsqueeze(2).expand(f.size(0),f.size(1),y.size(2)).contiguous();
        out = {}
        out['y'] = yout
        if self.mode.startswith('Inv'):
            inv_y = torch.cat((y,expf),1).contiguous();
            inv_y = self.inv_decoder[0](inv_y);
            inv_y = inv_y.transpose(2,1).contiguous();
            out['inv_x'] = inv_y;
        if self.grid_num > 1:
            out['grid_x'] = torch.cat([grid  for i in range(0,self.grid_num)],2).contiguous().transpose(2,1).contiguous(); 
        else:
            out['grid_x'] = grid.transpose(2,1).contiguous();
        #print(out['grid_x'].shape);
        return out;