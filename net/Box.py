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
class BoxNet(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=4,fc=False);
        self.dec_size = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(256, 3, kernel_size=1, bias=True),
                nn.Sigmoid()
                );
        self.dec_rot6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(256, 6, kernel_size=1, bias=True),
                nn.Tanh()
                );
                
    def forward(self,x):
        x = self.enc(x);
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
        x1,x2 = self.add_msk(x,ms,mt);
        #
        ss1,sr1,sr2 = self.bnet(x1);
        sb = self.sr2box(ss1,sr1,sr2);
        #
        ts1,tr1,tr2 = self.bnet(x2);
        tb = self.sr2box(ts1,tr1,tr2);
        #
        out = {'sb':sb,'tb':tb,'ss':ss,'sr1':sr1,'sr2':sr2,'ts':ts,'tr1':tr1,'tr2':tr2};
        return out;
        
    def rot(self,x_raw,y_raw):
            x = F.normalize(x_raw,dim=1,p=2);
            z = torch.cross(x,y_raw);
            z = F.normalize(z,dim=1,p=2);
            y = torch.cross(z,x);
            rot = torch.stack([x,y,z],dim=1);
            rot = rot.view(-1,3,3);
        return rot;
        
    def add_msk(self,x,ms,mt):
        if self.mode == 'full':
            x1 = torch.cat([x,ms],axis=1);
            x2 = torch.cat([x,mt],axis=1);
        elif self.mode == 'part':
            x1 = torch.cat([x*ms,ms],axis=1);
            x2 = torch.cat([x*mt,mt],axis=1);
        else:
            assert False, "Unkown mode";
        return x1,x2;
        
    def sr2box(self,size,r1,r2):
        const = np.array([[[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]]],dtype=np.float32);
        const = torch.from_numpy(const);
        const = const.type(size.type());
        const = const.requires_grad = True;
        rot = self.rot(r1,r2);
        box = const*( size.unsqueeze(1).contiguous() );
        box = torch.matmul(box,rot);
        return box;