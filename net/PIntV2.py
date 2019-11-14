import torch;
import numpy as np;
import torch.nn as nn;
from .upart import *;
#
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 32);
        self.down1 = Down(32, 64);
        self.down2 = Down(64, 128);
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.upu1 = Up(512, 128, bilinear)
        self.upu2 = Up(256, 64, bilinear)
        self.upu3 = Up(128, 32, bilinear)
        self.upu4 = Up(64, 32, bilinear)
        self.outu = OutConv(32, n_classes//2)
        self.upv1 = Up(512, 128, bilinear)
        self.upv2 = Up(256, 64, bilinear)
        self.upv3 = Up(128, 32, bilinear)
        self.upv4 = Up(64, 32, bilinear)
        self.outv = OutConv(32, n_classes//2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u = self.upu1(x5, x4)
        u = self.upu2(u, x3)
        u = self.upu3(u, x2)
        u = self.upu4(u, x1)
        u = self.outu(u)
        v = self.upv1(x5, x4)
        v = self.upv2(v, x3)
        v = self.upv3(v, x2)
        v = self.upv4(v, x1)
        v = self.outv(v)
        logits = torch.cat([u,v],dim=1).contiguous();
        return logits

# map (b,6,224,224)
# p (b,100)
# dp(b,2,100)
def path_integral(map,p,dp):
    map = map.view(map.size(0),6,-1);#(b,6,224x224)
    p = p.unsqueeze(1).repeat(1,6,1).contiguous();#(b,6,num)
    v = torch.gather(map,2,p.long());#(b,6,num)
    v = v.view(v.size(0),2,3,-1);
    dp = dp.view(dp.size(0),2,1,-1);
    dp = dp.type(v.type());
    return torch.sum(v*dp,dim=[1,3]);
#
def path_generate(s,e,h,w,num=200):
    return line(s,e,h,w,num);
#    
def line(s,e,h,w,num):
    b = s.size(0);
    t = torch.linspace(0.0,1.0,num);
    t = t.view(1,num);
    t = t.type(s.type());
    x = t*s[:,:1] + (1.0-t)*e[:,:1];
    y = t*s[:,1:] + (1.0-t)*e[:,1:];
    p = y*w + x;
    dp = ( e - s ) / float(num);
    dp = dp.view( dp.size(0), dp.size(1), 1 );
    dp = dp.repeat(1,1,num);
    return p,dp;
#
def bspline(start,end):
    return;
#    
class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.unet = UNet(3,6);
        
    def forward(self,input):
        img = input[0];
        s = input[6];
        e = input[7];
        x = img[:,:,:,:3].contiguous();
        x = x.permute(0,3,1,2).contiguous();
        dmap = self.unet(x);
        p,dp = path_generate(s,e,img.size(2),img.size(3));
        y = path_integral(dmap,p,dp);
        out = {'vec':y};
        out['y'] = self.v2a(y);
        out['dmap'] = dmap;
        return out;
        
    def v2a(self,vec):
        r = torch.sqrt(torch.sum((vec[:,:3].contiguous())**2,dim=1));
        theta = torch.acos(vec[:,2].contiguous()/r) / np.pi;
        phi = ( torch.atan2(vec[:,1].contiguous(),vec[:,0].contiguous()) + np.pi )/2.0/np.pi;
        rt = torch.stack([theta.view(-1),phi.view(-1)],dim=1);
        return rt;
        
def run(**kwargs):
    import matplotlib.pyplot as plt;
    from PIL import Image;
    im = Image.open("./data/exp.png",'r');
    pim = np.array(im);
    im = torch.from_numpy(pim);
    print(im.size());
    im = im.permute(2,0,1).contiguous();
    print(im.size());
    x = im[0,:,:].numpy();
    print(x.shape);
    im = Image.fromarray(x,'L');
    im.save("./log/out.png");
    return;