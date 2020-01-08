import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;
from .cageutil import add_msk_dual as add_msk
        
class TouchNet(nn.Module):
    def __init__(self,**kwargs):
        super(TouchNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=4,fc=False);
        self.dec = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        );
        self.avgpool = nn.AvgPool2d(7);
        self.fc = nn.Linear(512,1);
                
    def forward(self,x1,x2):
        #
        x1 = self.enc(x1);
        x2 = self.enc(x2);
        x = torch.cat([x1,x2],axis=1);
        x = self.dec(x);
        #
        x = self.avgpool(x);
        x = x.view(x.size(0), -1);
        x = self.fc(x);
        y = F.sigmoid(x);
        return y;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.mode = kwargs['mode'];
        self.tnet = TouchNet();
        
    def forward(self,input):
        img = input[0];
        x = img[:,:,:,:3].contiguous();
        x = x.permute(0,3,1,2).contiguous();
        #
        ms = input[1].unsqueeze(1);
        mt = input[2].unsqueeze(1);
        #
        x1,x2 = add_msk(self,x,ms,mt);
        y = self.tnet(x1,x2);
        #
        out = {'y':y};
        return out;
            
        
        
        
        