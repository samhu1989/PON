import torch.nn as nn;
import torch.nn.functional as F;
import torch;
import numpy as np;
from ..resnet import resnet18;
from .box import Box;
from .box import box_face;

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers;

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.mode = kwargs['mode'];
        if  self.mode == 'GAN':
            self.im_enc = resnet18(False,input_channel=7,num_classes=512);
        else:
            self.im_enc = resnet18(False,input_channel=7,num_classes=1024);
        self.box = Box(**kwargs);
        self.gen_f = nn.Sequential(
            *block(1024,1024, normalize=False),
            *block(1024,512)
        );
        self.gen_s = nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,3),
            nn.Hardtanh(min_val=0.1,max_val=10.0,inplace=True);
        );
        self.gen_r = nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,4),
            nn.Hardtanh(min_val=-1.0,max_val=1.0,inplace=True);
        );
        self.gen_t = nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,3),
            nn.Hardtanh(min_val=-2.0,max_val=2.0,inplace=True);
        );
        self._init_layers();
        
    def forward(self,input):
        img = input[0][:,:3,:,:].contiguous();
        src = input[1].unsqueeze(1);
        tgt = input[2].unsqueeze(1);
        coord = torch.from_numpy((np.mgrid[-1:1:224j,-1:1:224j]).astype(np.float32));
        coord = coord.unsqueeze(0).expand(img.size(0),coord.size(0),coord.size(1),coord.size(2));
        coord = coord.type(img.type());
        coord = coord.requires_grad_(True);
        x = torch.cat((img,src,tgt,coord),1).contiguous();
        src_box = input[3];
        fimg = self.im_enc(x);
        if self.mode == 'GAN':
            noise = input[-1];
            gen_input = torch.cat((fimg,noise), 1);
        else:
            gen_input = fimg;
        f = self.gen_f(gen_input);
        s = self.gen_s(f);
        r = self.gen_r(f);
        if not self.training:
           r = r / torch.sqrt(torch.sum(r**2,dim=1,keepdim=True));
        t = self.gen_t(f);
        bs,pts = self.box(s,r,t,src_box);
        out = {};
        out['box'] = bs.contiguous();
        out['pts'] = pts.transpose(2,1).contiguous();
        out['rot'] = r.contiguous();
        out['grid_x'] = (torch.from_numpy(box_face)).unsqueeze(0).expand(img.size(0),box_face.shape[0],box_face.shape[1]);
        out['y'] = out['box'];
        return out;
        
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

