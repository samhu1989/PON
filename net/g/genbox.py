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
        layers.append(nn.BatchNorm1d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers;
    
def printbn(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward');
    mean = input[0].mean(dim=0);
    var = input[0].var(dim=0);
    p = self.running_mean.data.cpu().numpy();
    print("self.istraining",self.training);
    print("running mean",p[0]);
    print("input mean",mean.data.cpu().numpy()[0]);
    

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
            *block(1024,1024,False),
            *block(1024,512)
        );
        self.gen_sr = nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,9),
            nn.Hardtanh(min_val=-5.0,max_val=5.0,inplace=True)
        );
        self.gen_t = nn.Sequential(
            *block(512,256),
            *block(256,128),
            nn.Linear(128,3),
            nn.Hardtanh(min_val=-2.0,max_val=2.0,inplace=True)
        );
        self._init_layers();
        #self.print_info();
        
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
        sr = self.gen_sr(f);
        t = self.gen_t(f);
        bs,pts = self.box(sr,t,src_box);
        out = {};
        out['box'] = bs.contiguous();
        out['pts'] = pts.transpose(2,1).contiguous();
        out['sr'] = sr;
        out['t'] = t;
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
                
    def print_info(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm1d):
                m.register_forward_hook(printbn)
                break;
        

