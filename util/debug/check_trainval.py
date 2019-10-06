import traceback
import importlib
import sys;
sys.path.append('./ext/');
import cd.dist_chamfer as ext;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from ..data.ply import read_ply;
torch.backends.cudnn.enabled = False;
distChamfer = ext.chamferDist();

data = read_ply('./data/optimize/bunny.ply');
pts = np.array(data['points'])
pts = pts[np.random.choice(pts.shape[0],2500),:];
pts = torch.from_numpy(pts.transpose(1,0).reshape(1,3,-1));
pts = pts.cuda();
pts.requires_grad = True;
var = torch.randn((1,2500,1),requires_grad=True);
optimizer = optim.Adam([var],lr=1e-3,weight_decay=0);
fig = plt.figure();
ax = p3.Axes3D(fig);

g = pts.data.cpu().numpy().astype(np.float32);
v = var.data.cpu().numpy().astype(np.float32);
pg = ax.scatter(g[0,0,...],g[0,1,...],g[0,2,...],c='r', marker='o');
pv = ax.scatter([],[],[],c='b', marker='*');
pv.set_offset_position('screen');

def init():
    return [pv];

# animation function.  This is called sequentially
def loss1(x,y):
    dist1, dist2 = distChamfer(x,y);
    return torch.mean(dist1)+torch.mean(dist2);
    
def loss2(x,y):
    return torch.mean((x - y)**2);

def animate(i,net,optimizer):
    varcuda = var.cuda();
    net.train();
    out = net([varcuda]);
    ygt = pts.transpose(2,1).contiguous();
    loss = loss1(ygt,out['y']);
    #print(loss.data.cpu().numpy());
    #print(out['y'].data.cpu().numpy());
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    print(loss.data.cpu().numpy());
    net.eval();
    with torch.no_grad():
        out = net([varcuda]);
    v = out['y'].data.cpu().numpy().astype(np.float32);
    pv._offsets3d = (v[0,...,0],v[0,...,1],v[0,...,2]);
    return [pv]

def run(**kwargs):
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        for k in kwargs.keys():
            if not kwargs[k] is None:
                opt[k] = kwargs[k];
        opt['pts_num'] = 2500;
        opt['pts_num_gt'] = 2500;
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        print(opt['mode'])
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
        if torch.cuda.is_available():
            net = net.cuda();
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #run the code
    param = [var];
    param.extend(net.parameters());
    print(net.parameters())
    optimizer = eval('optim.'+opt['optim'])(param,lr=opt['lr'],weight_decay=opt['weight_decay']);
    animfunc = partial(animate,net=net,optimizer=optimizer);
    anim = animation.FuncAnimation(fig, animfunc, init_func=init,frames=200, interval=20, blit=False)
    plt.show();