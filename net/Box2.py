import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;
#
import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;
#
from .cageutil import sr2box;
from .cageutil import add_msk_dual as add_msk;

class BoxNet(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.dec_size = nn.Sequential(
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 3, kernel_size=1, bias=True)
                );
        self.dec_rot6 = nn.Sequential(
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
                nn.Conv2d(512, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 6, kernel_size=1, bias=True)
                );
                
    def forward(self,x):
        size = self.dec_size(x);
        size = size.view(size.size(0),-1);
        rot6 = self.dec_rot6(x);
        rot6 = rot6.view(rot6.size(0),6);
        r1 = rot6[:,0:3].contiguous();
        r2 = rot6[:,3:6].contiguous();
        return size,r1,r2;
#
class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.mode = kwargs['mode'];
        self.bnet = BoxNet();
        
    def forward(self,input):
        img = input[0];
        x = img[:,:,:,:3].contiguous();
        x = x.permute(0,3,1,2).contiguous();
        #
        ms = input[1].unsqueeze(1);
        mt = input[2].unsqueeze(1);
        #
        x1,x2 = add_msk(self,x,ms,mt);
        #
        ss,sr1,sr2 = self.bnet(x1);
        sb = sr2box(ss,sr1,sr2);
        #
        ts,tr1,tr2 = self.bnet(x2);
        tb = sr2box(ts,tr1,tr2);
        #
        out = {'sb':sb,'tb':tb,'ss':ss,'sr1':sr1,'sr2':sr2,'ts':ts,'tr1':tr1,'tr2':tr2};
        return out;
        
