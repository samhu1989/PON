import torch.nn as nn;
import torch.nn.functional as F;
import torch;
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
            nn.Linear(256,1),
            nn.Sigmoid()
            );

    def forward(self,input):
        img = input[0];
        pts = input[1];
        fim = self.im_enc(img);
        fpt = self.pt_enc(pts);
        f = torch.cat([fim,fpt],-1);
        y = self.cls(f)
        return y;