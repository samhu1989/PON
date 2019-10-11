import torch;
import torch.nn as nn;
import numpy as np;
from ..quat import qrot;
from util.sample import tri2pts_batch;
bv = np.array(
    [
        [-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0],
        [-0.5,0.5,1.0],[0.5,0.5,1.0],[0.5,-0.5,1.0],[-0.5,-0.5,1.0]
    ],
    dtype=np.float32
)
box_face = np.array(
[
[1,3,0],
[2,3,1],
[7,5,4],
[7,6,5],
[4,5,0],
[5,1,0],
[1,5,2],
[2,5,6],
[6,3,2],
[6,7,3],
[4,0,3],
[3,7,4]
],
dtype=np.int32
);

bv_th = torch.from_numpy(bv).view(1,8,3);

class Box(nn.Module):
    def __init__(self,**kwargs):
        super(Box,self).__init__();
        self.pts_num = kwargs['pts_num'];
        self.grid_num = kwargs['grid_num'];

    def forward(self,s,r,t,bv=bv_th):
        bv = bv.type(s.type());
        s = s.view(s.size(0),1,s.size(1));
        bv = s*bv;
        bv = bv.expand(s.size(0),8,3).contiguous();
        r  = r.unsqueeze(1).expand(r.size(0),8,4).contiguous();
        bv = qrot(r,bv);
        t = t.view(t.size(0),1,t.size(1));
        bv = bv + t;
        bpts = tri2pts_batch(bv,box_face,self.pts_num//box_face.shape[0]//self.grid_num);
        return bv,bpts;
        
def run(**kwargs):
    from .box import bv;
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import axes3d as p3;
    import pandas as pd;
    from util.data.ply import write_ply;
    fig = plt.figure();
    ax = fig.add_subplot(111,projection='3d');
    ax.axis('equal')
    bv = torch.from_numpy(bv).view(1,8,3);
    bv = bv.expand(2,8,3).contiguous();
    r = torch.randn([2,4]);
    r = r / torch.sqrt(torch.sum(r**2,dim=1,keepdim=True));
    r  = r.unsqueeze(1).expand(2,8,4).contiguous();
    bv = qrot(r,bv);
    ax.scatter(bv[0,...,0],bv[0,...,1],bv[0,...,2]);
    fidx = box_face;
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for fi in range(fidx.shape[0]):
        face[fi] = (3,fidx[fi,0],fidx[fi,1],fidx[fi,2]);
    write_ply('./a.ply',points = pd.DataFrame(bv[0,...].cpu().numpy()),faces=pd.DataFrame(face));
    plt.show();
        
        