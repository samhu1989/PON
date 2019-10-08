import torch.nn as nn;
import torch.nn.functional as F;
import torch;
from ..resnet import resnet18;
from .box import Box;

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.box_num = kwargs['grid_num'];
        self.im_enc = resnet18(False,num_classes=512);
        self.box();
        self.gen_f = nn.Sequential(
            *block(1024,1024, normalize=False),
            *block(1024,512)
        );
        self.gen_s = nn.ModuleList([nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,3),
            nn.Hardtanh(min_val=0.001,max_val=1.0,inplace=True)
        ) for i in range(self.box_num)]);
        self.gen_r = nn.ModuleList([nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,4),
            nn.Hardtanh(min_val=-1.0,max_val=1.0,inplace=True)
        ) for i in range(self.box_num)]);
        self.gen_t = nn.Sequential([nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,3)
            nn.Hardtanh(min_val=-1.5,max_val=1.5,inplace=True)
        ) for i in range(self.box_num)]);
        
    def forward(self,input):
        img = input[0];
        noise = input[-1];
        gen_input = torch.cat((self.im_enc(img),noise), 1);
        f = self.gen_f(gen_input);
        y = []
        for i in range(self.box_num):
            s = self.gen_s[i](f);
            r = self.gen_r[i](f);
            t = self.gen_t[i](f);
            y.append(self.box(s,r,t));
        yout = torch.cat(y,1);
        return yout;

