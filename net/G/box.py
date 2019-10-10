import torch.nn as nn;
import numpy as np;
from ..quat import qrot;
bv = np.array(
    [
        [-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0],
        [-0.5,0.5,1.0],[0.5,0.5,1.0],[0.5,-0.5,1.0],[-0.5,-0.5,1.0]
    ],
    dtype=np.float32
)
widx = [
[0,1,2,3],
[0,1,4,5],
[1,2,6,5],
[2,3,7,6],
[0,3,4,7],
[4,5,6,7]
];
def randw(n):
    rw = np.zeros([1,n,8,1],dtype=np.float32);
    for i in range(6):
        w = np.random.uniform([1,n//6,4,1],dtype=np.float32);
        w = w / np.sum(w,axis=2,keepdims=True);
        rw[:,:,widx[i],:] = w;
    return torch.from_numpy(rw);

class Box(nn.Module):
    def __init__(self,**kwargs):
        super(Box,self).__init__();
        self.pts_num = kwargs['pts_num'];
        self.grid_num = kwargs['grid_num'];
        self.box_vert = torch.from_numpy(bv).view(1,8,3);

    def forward(self,s,r,t):
        bv = self.box_vert.type(s.type());
        s = s.view(s.size(0),1,s.size(1));
        bv = s*bv;
        bv = bv.expand(s.size(0),8,3);
        r  = r.expand(r.size(0),8,4);
        bv = qrot(r,bv);
        t = t.view(t.size(0),1,t.size(1));
        bv = bv + t;
        bv = bv.view(bv.size(0),1,8,3);
        w = randw(pts_num//grid_num);
        w = w.type(bv.type());
        bv = torch.mean(bv*w,dim=2);
        return bv;
        
        