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
#
class BoxNet(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.conv_size = nn.Sequential(
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
                nn.Conv2d(512, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7)
                );
        self.fc_size = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2)
                );
        self.conv_rot6 = nn.Sequential(
                resnet.resnet18(pretrained=False,input_channel=4,fc=False),
                nn.Conv2d(512, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7)
                );
        self.fc_rot6 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 6),
                nn.Tanh()
                )
                
    def forward(self,x):
        size = self.conv_size(x);
        size = size.view(size.size(0),-1);
        size = self.fc_size(size);
        size = torch.abs(size);
        consts = torch.ones([size.size(0),1],requires_grad=True);
        consts = consts.type(size.type());
        size = torch.cat([consts,size],dim=1);
        rot6 = self.conv_rot6(x);
        rot6 = rot6.view(rot6.size(0),-1);
        rot6 = self.fc_rot6(rot6);
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
