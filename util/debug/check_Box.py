import traceback
import importlib
import sys;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation;
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from torch.utils.data import DataLoader;
from ..data.gen_toybox import box_face;
from net.g.box import Box;
from util.dataset.ToyV import recon;

#fig2d = plt.figure(figsize=(72,72));
fig3d = plt.figure(figsize=(54,72));
pv = [];
iternum = 300;


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);

def init():
    return pv;
    
    
def rot(r1,r2):
    rr1 = r1 / np.sqrt(np.sum(r1**2));
    rr2 = r2 - np.sum(r2*rr1)*rr1;
    rr2 = rr2 / np.sqrt(np.sum(rr2**2));
    r3 = np.cross(rr1,rr2);
    rot = np.stack([rr1,rr2,r3],axis=0);
    return rot;
    
    
def parse(vec):
    coord = np.array([[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]],dtype=np.float32);
    ss = vec[:3];
    coords = ss[np.newaxis,:]*coord;
    sr1 = vec[3:6];
    sr2 = vec[6:9]
    srot = rot(sr1,sr2);
    vs = np.dot(coords,srot.reshape(3,3))
    ts = vec[9:12];
    coordt = ts[np.newaxis,:]*coord;
    center = vec[12:15];
    tr1 = vec[15:18];
    tr2 = vec[18:21]
    trot = rot(tr1,tr2);
    vt = np.dot(coordt,trot.reshape(3,3)) + center[np.newaxis,:];
    return  vs,vt;

def animate(i,config,net,optim,data):
    print(i,'/',iternum);
    net.train();
    out = net(data);
    loss = config.loss(data,out);
    optim.zero_grad();
    loss['overall'].backward();
    optim.step();
    net.eval();
    with torch.no_grad():
        out = net(data);
    acc = config.accuracy(data,out);
    for k,v in acc.items():
        print(k,':',v);
    img = data[0].data.cpu().numpy();
    row = img.shape[0];
    vout = out['vec'].data.cpu().numpy();
    xms = np.transpose(out['xms'].data.cpu().numpy(),(0,2,3,1));
    xmt = np.transpose(out['xmt'].data.cpu().numpy(),(0,2,3,1));
    xmst = np.transpose(out['xmst'].data.cpu().numpy(),(0,2,3,1));
    ygt = data[3].data.cpu().numpy();
    vgt = data[4].data.cpu().numpy();
    yout = out['y'].data.cpu().numpy();
    col = 8
    for ri in range(row):
        #pv[ri*col].set_array(xms[ri,...]);
        #pv[ri*col+1].set_array(xmt[ri,...]);
        #pv[ri*col+2].set_array(xmst[ri,...]);
        ind = np.zeros([2,1],np.float32);
        ind[0,0] = ygt[ri,...];
        ind[1,0] = yout[ri,...];
        pv[ri*col+3].set_array(ind);
        ptsa,ptsb = parse(vgt[ri,...]);
        #
        va = ptsa[box_face,:];
        tmp = va[:,1].copy();
        va[:,1] = va[:,2];
        va[:,2] = tmp;

        #print(va);
        #
        vb = ptsb[box_face,:];
        tmp = vb[:,1].copy();
        vb[:,1] = vb[:,2];
        vb[:,2] = tmp;
        #print(vb);
        #
        pv[ri*col+4].set_verts(va,False);
        pv[ri*col+5].set_verts(vb,False);
        ptsa,ptsb = parse(vout[ri,...]);
        #
        va = ptsa[box_face,:];
        tmp = va[:,1].copy();
        va[:,1] = va[:,2];
        va[:,2] = tmp;
        #print(va);
        #
        vb = ptsb[box_face,:];
        tmp = vb[:,1].copy();
        vb[:,1] = vb[:,2];
        vb[:,2] = tmp;
        #print(vb);
        #
        pv[ri*col+6].set_verts(va,False);
        pv[ri*col+7].set_verts(vb,False);
        #pv[ri*col+5].set_3d_properties(ptsb[:,1]);

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
        train_data = m.Data(opt,'train');
        val_data = m.Data(opt,opt['user_key']);
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
    msks = data[1].data.cpu().numpy();
    mskt = data[2].data.cpu().numpy();
    ygt = data[3].data.cpu().numpy();
    vgt = data[4].data.cpu().numpy();
    #run the code
    optim = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);
    tri = box_face;
    global pv;
    row = img.shape[0];
    col = 6;#img,msks,mskt,xmt,xms,xmst,ygt,yout,vgt,vout
    for ri in range(row):
        #ax1 = fig.add_subplot(row*2,col//2,ri*col+1);
        #ax1.imshow(img[ri,...]);
        ax4 = fig3d.add_subplot(row*2,col//2,ri*col+1);
        pv.append(ax4.imshow(img[ri,...]));
        ax5 = fig3d.add_subplot(row*2,col//2,ri*col+2);
        pv.append(ax5.imshow(msks[ri,...],cmap='gray'));
        ax6 = fig3d.add_subplot(row*2,col//2,ri*col+3);
        pv.append(ax6.imshow(mskt[ri,...],cmap='gray'));
        ind = np.zeros([2,1],np.float32);
        ind[0,0] = ygt[ri,...];
        ax7 = fig3d.add_subplot(row*2,col//2,ri*col+4);
        pv.append(ax7.imshow(ind,cmap='gray'));
        ax8 = fig3d.add_subplot(row*2,col//2,ri*col+5,projection='3d');
        ax8.view_init(elev=20, azim=90)
        ax8.set_aspect('equal', adjustable='box');
        ax8.set_xlim([-1,1]);
        ax8.set_ylim([-1,1]);
        ax8.set_zlim([-1,1]);
        ptsa,ptsb = parse(vgt[ri,...]);
        pv.append(ax8.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1)));
        pv.append(ax8.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1)));
        ax9 = fig3d.add_subplot(row*2,col//2,ri*col+6,projection='3d');
        ax9.view_init(elev=20, azim=90)
        ax9.set_aspect('equal', adjustable='box');
        ax9.set_xlim([-1,1]);
        ax9.set_ylim([-1,1]);
        ax9.set_zlim([-1,1]);
        ptsa,ptsb = parse(vgt[ri,...]);
        pv.append(ax9.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1)));
        pv.append(ax9.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1)));
        
        
    #run the code
    animfunc = partial(animate,config=config,net=net,optim=optim,data=data);
    anim = animation.FuncAnimation(fig3d, animfunc, init_func=init,frames=iternum, interval=30, blit=False);
    plt.show();

    writer = PillowWriter(fps=30, metadata=dict(artist='Siyu'));
    anim.save('./log/debug_Box.gif',writer=writer);
