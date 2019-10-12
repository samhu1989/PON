import traceback
import importlib
import sys;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from torch.utils.data import DataLoader;
from ..data.gen_toybox import box_face;
from net.g.box import Box;

fig = plt.figure(figsize=(64,32));
pv = [];
iternum = 250;


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);

def init():
    return pv;

def animate(i,config,net,optim,data):
    print(i,'/',iternum);
    net.train();
    out = net(data);
    loss = config.loss(data,out);
    optim.zero_grad();
    loss['overall'].backward();
    optim.step();
    print(loss['overall'].data.cpu().numpy());
    net.eval();
    with torch.no_grad():
        out = net(data);
    box_src = data[3].data.cpu().numpy();
    box_tgt = data[4].data.cpu().numpy();
    box_all = data[5].data.cpu().numpy();
    box_out = out['box'].data.cpu().numpy();
    num = box_src.shape[0];
    col = 8;
    row = num // col;
    for ri in range(row):
        for cj in range(col):
            ni = ri*col+cj;
            pv[ni].set_verts(box_out[ni,box_face,:],False);
    
    return pv;

def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        for k in kwargs.keys():
            if not kwargs[k] is None:
                opt[k] = kwargs[k];
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
        if torch.cuda.is_available():
            net = net.cuda();
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+opt['dataset']);
        train_data = m.Data(opt,True);
        val_data = m.Data(opt,False);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
        
    for i, data in enumerate(train_load,0):
        data2cuda(data);
        d = data;
        break;
    #
    box_src = data[3].data.cpu().numpy();
    box_tgt = data[4].data.cpu().numpy();
    box_all = data[5].data.cpu().numpy();
    #run the code
    optim = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);

    tri = box_face;
    global pv;
    num = box_src.shape[0];
    col = 8;
    row = num // col;
    for ri in range(row):
        for cj in range(col):
            ni = ri*col+cj;
            ax = fig.add_subplot(row,col,ni+1,projection='3d');
            ax.axis('equal');
            pv.append(ax.plot_trisurf(box_src[ni,...,0],box_src[ni,...,1],tri,box_src[ni,...,2],color=(1,0,0,1)));
            ax.plot_trisurf(box_tgt[ni,...,0],box_tgt[ni,...,1],tri,box_tgt[ni,...,2],color=(0,0,1,0.3));
            ax.plot_trisurf(box_src[ni,...,0],box_src[ni,...,1],tri,box_src[ni,...,2],color=(0,1,0,0.3));
    
    #run the code
    animfunc = partial(animate,config=config,net=net,optim=optim,data=data);
    anim = animation.FuncAnimation(fig, animfunc, init_func=init,frames=iternum, interval=30, blit=False);
    Writer = animation.writers['ffmpeg'];
    writer = Writer(fps=30, metadata=dict(artist='Siyu'));
    anim.save('./log/debug_genbox.mp4',writer=writer);