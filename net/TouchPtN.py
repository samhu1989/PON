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
from .cageutil import add_msk_dual as add_msk;
from .cageutil import sr2box;
#
def norm_size(s):
    sm, _ = torch.max(s,dim = 1,keepdim=True);
    return s / sm;

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
        self.w1 = nn.Sequential(
            nn.Conv1d(512+3, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1, bias=True),
            nn.Softmax(dim=2)
        );
        self.w2 = nn.Sequential(
            nn.Conv1d(512+3, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1, bias=True),
            nn.Softmax(dim=2)
        );
                
    def forward(self,x1,b1,x2,b2):
        #
        x1 = self.enc(x1);
        x2 = self.enc(x2);
        x = torch.cat([x1,x2],axis=1);
        x = self.dec(x);
        x = x.view(x.size(0),512,1);
        x = x.repeat(1,1,8);
        #
        f1 = torch.cat([x,b1.permute(0,2,1).contiguous()],dim=1);
        w1 = self.w1(f1);
        w1 = w1.permute(0,2,1).contiguous();
        #
        f2 = torch.cat([x,b2.permute(0,2,1).contiguous()],dim=1);
        w2 = self.w2(f2);
        w2 = w2.permute(0,2,1).contiguous();
        #
        #coords = torch.sum(w1*b1,dim=1);
        #coordt = torch.sum(w2*b2,dim=1);
        #
        return w1,w2;
        
class ScaleNet(nn.Module):
    def __init__(self,**kwargs):
        super(ScaleNet,self).__init__();
        self.enc = resnet.resnet18(pretrained=False,input_channel=4,fc=False);
        self.dec = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7)
        );
        self.fc = nn.Linear(512,1);
                
    def forward(self,x1,x2):
        #
        x1 = self.enc(x1);
        x2 = self.enc(x2);
        x = torch.cat([x1,x2],axis=1);
        x = self.dec(x);
        x = x.view(x.size(0), -1);
        scale = self.fc(x);
        #
        return scale;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.mode = kwargs['mode'];
        self.tpnet = TouchPtNet();
        self.snet = ScaleNet();
        
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
        vgt = input[4];
        #
        ss_gt = vgt[:,0:3].contiguous();
        sr1_gt = vgt[:,3:6].contiguous();
        sr2_gt = vgt[:,6:9].contiguous();
        sb_gt = sr2box(ss_gt,sr1_gt,sr2_gt);
        unit_sb_gt = sr2box(norm_size(ss_gt),sr1_gt,sr2_gt); 
        #
        ts_gt = vgt[:,9:12].contiguous();
        tr1_gt = vgt[:,15:18].contiguous();
        tr2_gt = vgt[:,18:21].contiguous();
        unit_tb_gt = sr2box(norm_size(ts_gt),tr1_gt,tr2_gt);
        #
        w1,w2 = self.tpnet(x1,unit_sb_gt,x2,unit_tb_gt);
        scale = self.snet(x1,x2);
        sbase, _ = torch.max(ss_gt,dim = 1,keepdim=True);
        ts_out = scale / ( sbase + np.finfo(np.float32).eps );
        tb_out = sr2box(norm_size(ts_gt)*ts_out,tr1_gt,tr2_gt);
        coords = torch.sum(w1*sb_gt,dim=1);
        coordt = torch.sum(w2*tb_out,dim=1);
        t = coords - coordt;
        tb = tb_out + t.unsqueeze(1).contiguous();
        out = {'t':t,'sb':sb_gt,'tb':tb,'ts':ts_out,'tr1':tr1_gt,'tr2':tr2_gt,'w1':w1,'w2':w2,'rs':scale};
        return out;