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
from util.dataset.ToyV import mv,mv_inv,sph2car,proj;

fig = plt.figure(figsize=(64,32));
pv = [];
iternum = 300;

def recon(center,r,angle):
    coord = np.concatenate([r.reshape(-1,1),angle.reshape(-1,2)],axis=1);
    c3dir = sph2car(coord.reshape(1,-1));
    c3d = np.zeros([2,3],dtype=np.float32);
    c = np.expand_dims(center[1,:3],0);
    c3d[1,:] = c[:,:3];
    c3d[0,:] = c3d[1,:] + c3dir;
    c3d = mv_inv(c3d);
    return c3d;



def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);
            
def norm_im(im):
    im = im - np.min(np.min(im,axis=0,keepdims=True),axis=1,keepdims=True);
    im = im / np.max(np.max(im,axis=0,keepdims=True),axis=1,keepdims=True);
    return im;

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
    box3d_src = data[1].data.cpu().numpy();
    box3d_tgt = data[2].data.cpu().numpy();
    center = data[3].data.cpu().numpy();
    r = data[4].data.cpu().numpy();
    gts = data[5].data.cpu().numpy();
    y = out['y'].data.cpu().numpy();
    map = out['dmap'].permute(0,2,3,1).data.cpu().numpy();
    
    num = box3d_src.shape[0];
    col = int(np.sqrt(num));
    row = num // col;
    for ri in range(row):
        for cj in range(col):
            ni = ri*col+cj;
            ymap = y[ni,...];
            ymap *= np.pi;
            ymap[1] *= 2;
            c3d = recon(center[ni,...],r[ni,...],ymap);
            c2d = proj(mv(c3d))
            pv[4*ni+0].set_data(c2d[:,0],c2d[:,1]);
            im1 = norm_im(map[ni,:,:,:3]);
            pv[4*ni+1].set_array(im1);
            im2 = norm_im(map[ni,:,:,3:6]);
            pv[4*ni+2].set_array(im2);
            pv[4*ni+3].set_data(c3d[:,0],c3d[:,1]);
            pv[4*ni+3].set_3d_properties(c3d[:,2]);
    if i == iternum-1:
        exit();
    return pv;

def run(**kwargs):
    global iternum;
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
    iternum = opt['nepoch']
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
    img = data[0].data.cpu().numpy();
    box_src = data[1].data.cpu().numpy();
    box_tgt = data[2].data.cpu().numpy();
    center = data[3].data.cpu().numpy();
    r = data[4].data.cpu().numpy();
    gt = data[5].data.cpu().numpy();
    #run the code
    optim = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);
    tri = box_face;
    global pv;
    num = box_src.shape[0];
    col = int(np.sqrt(num));
    row = num // col;
    for ri in range(row):
        for cj in range(col):
            ni = ri*col+cj;
            y = gt[ni,...].copy();
            y *= np.pi;
            y[1] *= 2;
            c3d = recon(center[ni,...],r[ni,...],y);
            #===========================================
            ax = fig.add_subplot(row,col*4,4*ni+1);
            ax.imshow(img[ni,...]);
            c2d = proj(mv(c3d));
            line = ax.plot(c2d[:,0],c2d[:,1],color='r');
            pv.extend(line);
            ax.plot(c2d[:,0],c2d[:,1],color='k');
            #===========================================
            ax = fig.add_subplot(row,col*4,4*ni+2);
            im1 = ax.imshow(img[ni,...]);
            pv.append(im1);
            #===========================================
            ax = fig.add_subplot(row,col*4,4*ni+3);
            im2 = ax.imshow(img[ni,...]);
            pv.append(im2);
            ax = fig.add_subplot(row,col*4,4*ni+4,projection='3d');
            ax.axis('equal');
            ax.plot_trisurf(box_tgt[ni,...,0],box_tgt[ni,...,1],tri,box_tgt[ni,...,2],color=(0,0,1,0.1));
            ax.plot_trisurf(box_src[ni,...,0],box_src[ni,...,1],tri,box_src[ni,...,2],color=(0,1,0,0.1));
            line = ax.plot(c3d[:,0],c3d[:,1],c3d[:,2],color='r');
            pv.extend(line);
            ax.plot(c3d[:,0],c3d[:,1],c3d[:,2],color='k');
    #run the code
    animfunc = partial(animate,config=config,net=net,optim=optim,data=data);
    anim = animation.FuncAnimation(fig, animfunc, init_func=init,frames=iternum, interval=30, blit=False);
    plt.show();
    '''
    Writer = animation.writers['ffmpeg'];
    writer = Writer(fps=30, metadata=dict(artist='Siyu'));
    anim.save('./log/debug_VReg.mp4',writer=writer);
    '''