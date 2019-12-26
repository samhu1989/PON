import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import net.resnet as resnet;
import numpy as np;

class BoxNet(nn.Module):
    def __init__(self,**kwargs):
        super(BoxNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=3,num_classes=512);
        self.dec1 = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(inplace=True)
                );
        self.dec_size = nn.Sequential(
                nn.Linear(512,3),
                nn.ReLU(inplace=True)
                );
        self.dec_r1 = nn.Linear(512,3);
        self.dec_r2 = nn.Linear(512,3);
                
    def forward(self,img):
        y = self.enc(img);
        y = self.dec1(y);
        size = self.dec_size(y);
        r1 = self.dec_r1(y);
        r2 = self.dec_r2(y);
        return size,r1,r2;
        
class CNet(nn.Module):
    def __init__(self,**kwargs):
        super(CNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=5,num_classes=512);
        self.dec_is = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(inplace=True),
                nn.Linear(512,1),
                nn.Sigmoid()
                );
        self.dec_x1 = nn.Sequential(
                nn.Conv1d(512+3,64,1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64,1,1)
                );
        self.dec_x2 = nn.Sequential(
                nn.Conv1d(512+3,64,1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64,1,1)
                );
                
    def forward(self,img,c1,c2):
        x = self.enc(img);
        y = self.dec_is(x);
        f = x.unsqueeze(2).repeat(1,1,c1.size(2)).contiguous();
        expf1 = torch.cat((c1,f),1).contiguous();
        expf2 = torch.cat((c2,f),1).contiguous();
        w1 = self.dec_x1(expf1);
        w1 = F.softmax(w1,dim=2);
        w2 = self.dec_x2(expf2);
        w2 = F.softmax(w2,dim=2);
        return y,w1,w2;

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
        srot = self.rot(sr1,sr2);
        c1 = torch.matmul(const,srot);
        c1 = c1.permute(0,2,1).contiguous();
        #
        ts,tr1,tr2 = self.bnet(xmt);
        trot = self.rot(tr1,tr2);
        c2 = torch.matmul(const,trot);
        c2 = c2.permute(0,2,1).contiguous();
        #
        xmst = x*(ms+mt);
        xmst_ms_mt = torch.cat([xmst,ms,mt],dim=1);
        y,ws,wt = self.cnet(xmst_ms_mt,c1,c2);
        #
        coords = torch.matmul(const*ss.unsqueeze(1).contiguous(),srot);
        coords = coords.permute(0,2,1).contiguous();
        coords = torch.sum(ws*coords,dim=2);
        coordt = torch.matmul(const*ts.unsqueeze(1).contiguous(),trot);
        coordt = coordt.permute(0,2,1).contiguous();
        coordt = torch.sum(wt*coordt,dim=2);
        vec = torch.cat([ss,sr1,sr2,ts,coords-coordt,tr1,tr2],dim=1);
        out = {'xms':xms,'xmt':xmt,'xmst':xmst,'y':y,'vec':vec,'xs':coords,'xt':coordt};
        return out;
        
    def rot(self,r1,r2):
        if self.training:
            r3 = torch.cross(r1,r2);
            rot = torch.stack([r1,r2,r3],dim=1);
            rot = rot.view(-1,3,3);
        else:
            rr1 = r1 / torch.sqrt(torch.sum(r1**2,dim=1,keepdims=True));
            rr2 = r2 - torch.sum(r2*rr1,dim=1,keepdims=True)*rr1;
            rr2 = rr2 / torch.sqrt(torch.sum(rr2**2,dim=1,keepdims=True));
            r3 = torch.cross(rr1,rr2);
            rot = torch.stack([rr1,rr2,r3],dim=1);
            rot = rot.view(-1,3,3);
        return rot;
        
        
        