import torch.nn as nn;
import torch.nn.functional as F;
import torch;
import numpy as np;
from ..resnet import resnet18;
from ..AtlasNet import PointNetfeat;

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.box_num = kwargs['grid_num'];
        self.pts_num = kwargs['pts_num'];
        self.im_enc = resnet18(False,num_classes=512);
        self.pt_enc = nn.Sequential(
                PointNetfeat(self.pts_num, global_feat=True, trans = False),
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True)
                );
        self.cls = nn.Sequential(
            *block(1024,512),
            *block(512,256),
            nn.Linear(256,1)
            );
        self._init_layers();

    def forward(self,input):
        img = input[0][:,:3,:,:].contiguous();
        pts = input[1];
        fim = self.im_enc(img);
        fpt = self.pt_enc(pts);
        f = torch.cat([fim,fpt],-1);
        y = self.cls(f)
        return y;
        
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,np.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();