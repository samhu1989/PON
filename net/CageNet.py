import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;

class ImEnc(nn.Module):
    def __init__(self,**kwargs):
        super(ImEnc,self).__init__();
        #part instance encoder
        self.nenc = resnet.resnet18(pretrained=False,input_channel=4,fc=False);
        #part pair encoder
        self.penc = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512)
        );
        
    def forward(self,x,pi,pj):
        nf = self.nenc(x);
        pnfi = torch.index_select(nf,dim=0,pi);
        pnfj = torch.index_select(nf,dim=0,pj);
        pnf = torch.stack([pnfi,pnfj],dim=1);
        pf = self.penc(pnf);
        return nf,pf;

class BoxDec(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.dec_size = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(256, 256, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 3, kernel_size=1, bias=True),
                nn.Sigmoid()
                );
        self.dec_rot6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(256, 256, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 6, kernel_size=1, bias=True),
                nn.Tanh()
                );
                
    def forward(self,x):
        size = self.dec_size(x);
        size = size.view(size.size(0),-1);
        rot6 = self.dec_rot6(x);
        rot6 = rot6.view(rot6.size(0),6);
        r1 = rot6[:,0:3].contiguous();
        r2 = rot6[:,3:6].contiguous();
        return size,r1,r2;
    

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.cnet = CNet();
        self.bnet = BoxNet();
        
    def forward(self,input):

        
        
        