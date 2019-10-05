import sys;
sys.path.append('./ext/');
import cd.dist_chamfer as ext;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d as p3;
distChamfer = ext.chamferDist();

rand_grid = torch.FloatTensor(1,3,1000);
rand_grid.normal_(0.0,1.0);
rand_grid += 1e-10;
rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True);
rand_grid = rand_grid.cuda();
rand_grid.requires_grad = True;
var = torch.randn((1,3,1000),requires_grad=True);
optimizer = optim.Adam([var],lr=1e-3,weight_decay=0);
fig = plt.figure();
ax = p3.Axes3D(fig);

g = rand_grid.data.cpu().numpy().astype(np.float32);
v = var.data.cpu().numpy().astype(np.float32);
pg, = ax.plot(g[0,0,...],g[0,1,...],g[0,2,...],c='r', marker='o');
pv, = ax.plot(v[0,0,...],v[0,1,...],v[0,2,...],c='b', marker='^');

def init():
    pv.set_data(v[0,0,...],v[0,1,...])
    pv.set_3d_properties(v[0,2,...])
    return [pv];

# animation function.  This is called sequentially
def loss1(x,y):
    dist1, dist2 = distChamfer(x,y);
    return torch.mean(dist1)+torch.mean(dist2);
    
def loss2(x,y):
    return torch.mean((x - y)**2);

def animate(i):
    x = var.cuda().transpose(2,1).contiguous();
    y = rand_grid.transpose(2,1).contiguous();
    loss = loss1(y,x);
    print(loss.data.cpu().numpy());
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    v = var.data.cpu().numpy().astype(np.float32);
    pv.set_data(v[0,0,...],v[0,1,...])
    pv.set_3d_properties(v[0,2,...])
    return [pv]

def run(**kwargs):
    anim = animation.FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=True)
    plt.show();