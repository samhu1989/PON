import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;

import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;

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
                nn.Conv2d(256, 6, kernel_size=1, bias=True)
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

class TouchPtNet(nn.Module):
    def __init__(self,**kwargs):
        super(TouchPtNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=4,fc=False);
        self.dec = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7)
        );
        self.avgpool = nn.Conv2d(1024, 512, kernel_size=1, bias=False),
        self.fc = nn.Linear(512,1);
                
    def forward(self,x1,p1,x2,p2):
        #
        x1 = self.enc(x1);
        x2 = self.enc(x2);
        x = torch.cat([x1,x2],axis=1);
        x = self.dec(x);
        x = x.view(x.size(0),1,-1);
        x = x.repeat(1,16,1);
        #
        x = torch.cat([x,p1])
        #
        return y;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.cnet = CNet();
        self.bnet = BoxNet();
        
    def forward(self,input):
        img = input[0];
        x = img[:,:,:,:3].contiguous();
        x = x.permute(0,3,1,2).contiguous();
        #
        ms = input[1].unsqueeze(1);
        mt = input[2].unsqueeze(1);
        #
        xms = x*ms;
        xmt = x*mt;
        const = np.array([[[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]]],dtype=np.float32);
        const = torch.from_numpy(const);
        const = const.type(x.type());
        #
        ss,sr1,sr2 = self.bnet(xms);
        with torch.no_grad():
            srot = self.rot(sr1,sr2);
        c1 = torch.matmul(const,srot);
        c1 = c1.permute(0,2,1).contiguous();
        #
        ts,tr1,tr2 = self.bnet(xmt);
        with torch.no_grad():
            trot = self.rot(tr1,tr2);
        c2 = torch.matmul(const,trot);
        c2 = c2.permute(0,2,1).contiguous();
        #
        xmst = x*(ms+mt);
        xmst_ms_mt = torch.cat([xmst,ms,mt],dim=1);
        y,ws,wt = self.cnet(xmst_ms_mt,c1,c2);
        #
        with torch.no_grad():
            coords = torch.matmul(const*ss.unsqueeze(1).contiguous(),srot);
            coords = coords.permute(0,2,1).contiguous();
            coordt = torch.matmul(const*ts.unsqueeze(1).contiguous(),trot);
            coordt = coordt.permute(0,2,1).contiguous();
        #
        coords = torch.sum(ws*coords,dim=2);
        coordt = torch.sum(wt*coordt,dim=2);
        vec = torch.cat([ss,sr1,sr2,ts,coords-coordt,tr1,tr2],dim=1);
        out = {'xms':xms,'xmt':xmt,'xmst':xmst,'y':y,'vec':vec,'xs':coords,'xt':coordt};
        return out;
        
    def rot(self,x_raw,y_raw):
            x = F.normalize(x_raw,dim=1,p=2);
            z = torch.cross(x,y_raw);
            z = F.normalize(z,dim=1,p=2);
            y = torch.cross(z,x);
            rot = torch.stack([x,y,z],dim=1);
            rot = rot.view(-1,3,3);
        return rot;