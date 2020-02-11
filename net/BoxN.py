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
from .cageutil import add_msk_sep as add_msk;

class BoxNet(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.dec_size = nn.Sequential(
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
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
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
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
#
class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.mode = kwargs['mode'];
        self.bnet = BoxNet();
        
    def forward(self,input):
        imgs = input[0];
        xs = imgs[:,:,:,:3].contiguous();
        xs = xs.permute(0,3,1,2).contiguous();
        ms = input[1].unsqueeze(1);
        #
        imgt = input[2];
        xt = imgt[:,:,:,:3].contiguous();
        xt = xt.permute(0,3,1,2).contiguous();
        mt = input[3].unsqueeze(1);
        #
        xs = add_msk(self,xs,ms);
        xt = add_msk(self,xt,mt);
        #
        ss,sr1,sr2 = self.bnet(xs);
        ts,tr1,tr2 = self.bnet(xt);
        if not self.training:
            ssm, _ = torch.max(ss,dim = 1,keepdim=True);
            ss = ss / ssm;
            tsm, _ = torch.max(ts,dim = 1,keepdim=True);
            ts = ts / tsm;
        #
        sb = sr2box(ss,sr1,sr2);
        tb = sr2box(ts,tr1,tr2);
        #
        out = {'sb':sb,'tb':tb,'ss':ss,'sr1':sr1,'sr2':sr2,'ts':ts,'tr1':tr1,'tr2':tr2};
        return out;
        
