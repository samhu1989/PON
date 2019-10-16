from __future__ import print_function
from PIL import Image
#sys import
import os;
import random;
import h5py;
import numpy as np;
from scipy import ndimage;
#torch import
import torch;
import torch.utils.data as data;
from ..data.ply import read_ply;
import pandas as pd;
import traceback;
from ..sample import tri2pts
import json;
from numpy.linalg import inv;
#project import
#
mvm = np.array(
[[1.0000, 0.0000,  0.0000, 0.0000],
[0.0000, -0.6000,  0.8000, 0.0000],
[0.0000,  0.8000,  0.6000, -2.5000],
[0.0000,  0.0000, 0.0000, 1.0000]]
 );

projm = np.array(
[[ 2.1875,0.,0.,0.],
 [ 0.,2.1875,0.,0.],
 [ 0.,0.,-1.002002,-0.2002002],
 [ 0.,0.,-1.,0.]
 ]
 );
 
def proj(pts):
    if pts.shape[-1] == 3:
        tmp = np.concatenate([pts,np.ones([pts.shape[0],1])],axis=1);
    elif pts.shape[-1] == 4:
        tmp = pts;
    else:
        assert False,'invalid input'; 
    tmp = tmp.transpose(1,0);
    tmp = np.matmul(projm,tmp);
    tmp = tmp.transpose(1,0);
    pts = tmp[:,:2];
    pts[:,0] /= tmp[:,3];
    pts[:,1] /= tmp[:,3];
    pts[:,0] = (224-1) * (1 - pts[:,0]) / 2;
    pts[:,1] = (224-1) * (pts[:,1] - 1) / (-2);
    return pts[:,:2];
    
def mv(pts):
    tmp = np.concatenate([pts,np.ones([pts.shape[0],1])],axis=1);
    tmp = tmp.transpose(1,0);
    tmp = np.matmul(mvm,tmp);
    pts = tmp.transpose(1,0);
    return pts[:,:3];
        
def mv_inv(pts):
    tmp = np.concatenate([pts,np.ones([pts.shape[0],1])],axis=1);
    tmp = tmp.transpose(1,0);
    tmp = np.matmul(inv(mvm),tmp);
    pts = tmp.transpose(1,0);
    return pts[:,:3];
    
def car2sph(vec):
    r = np.sqrt(np.sum(np.square(vec[:,:3]),axis=1));
    theta = np.arccos(vec[:,2]/r);
    phi = np.arctan2(vec[:,1],vec[:,0]) + np.pi;
    return np.stack([r,theta,phi],axis=1).astype(np.float32);
    
def sph2car(vec):
    coord = vec.copy();
    r = vec[:,0];
    th = vec[:,1];
    phi = vec[:,2];
    sth = np.sin(th);
    cth = np.cos(th);
    sphi = np.sin(phi - np.pi);
    cphi = np.cos(phi - np.pi);
    coord[:,0] = r*sth*cphi;
    coord[:,1] = r*sth*sphi;
    coord[:,2] = r*cth;
    return coord; 
 

class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.root = opt['data_path'];
        self.pts_num = opt['pts_num_gt'];
        self.train = train;
        self.datapath = [];
        if self.train:
            dataroot = os.path.join(self.root,'train');
        else:
            dataroot = os.path.join(self.root,'test');
        fs = os.listdir(dataroot);
        for f in fs:
            if (not 'msk' in f) and f.endswith('.json'):
                self.datapath.append(os.path.join(dataroot,f));
                
    def __getitem__(self, index):
        try:
            return self.load(index);
        except Exception as e:
            print(e);
            traceback.print_exc();
            exit();
        
    def load(self,idx):
        index = idx%self.__len__();
        fname = self.datapath[index];
        data = json.load(open(fname,'r'));
        num = len(data['box']);
        pick = np.random.randint(0,num);
        if pick == 0:
            srcpick = np.random.randint(1,num);
        else:
            if num > 2:
                alpha = np.random.randint(0,2)
                srcpick = alpha*np.random.randint(0,bspick)+(1-alpha)*np.random.randint(bspick+1,num);
            else:
                srcpick = 0;
        img = np.array(Image.open(fname.replace('.json','.ply_r_000.png'))).astype(np.float32)/255.0;
        img = img.astype(np.float32);
        s3d = np.array(data['box'][srcpick]);
        s3d = s3d.astype(np.float32);
        s2d = proj(mv(s3d));
        s2d = s2d.astype(np.float32);
        #
        t3d = np.array(data['box'][pick]);
        t3d = t3d.astype(np.float32);
        t2d = proj(mv(t3d));
        t2d = t2d.astype(np.float32);
        #
        c3d = np.concatenate([np.mean(t3d,axis=0,keepdims=True),np.mean(s3d,axis=0,keepdims=True)],axis=0);
        mvc3d = mv(c3d);
        dirc3d = mvc3d[0,:] - mvc3d[1,:];
        coord = car2sph(dirc3d.reshape(1,-1));
        coord = coord.astype(np.float32);
        #
        #s3d = self.mv(s3d);
        #t3d = self.mv(t3d);
        r = coord[:,0];
        gt = coord[:,1:3];
        gt[:,0] /= np.pi;
        gt[:,1] /= (2*np.pi); 
        return torch.from_numpy(),torch.from_numpy(s2d),torch.from_numpy(s3d),torch.from_numpy(t2d),torch.from_numpy(t3d),torch.from_numpy(r),torch.from_numpy(gt),'boxV';

    def __len__(self):
        return len(self.datapath);
        
#debuging the dataset      
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    opt['pts_num_gt'] = 1200;
    train_data = Data(opt,True);
    val_data = Data(opt,False);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    val_load = data.DataLoader(val_data,batch_size=1,shuffle=False,num_workers=opt['workers']);
    if not os.path.exists('./log/debug_dataset/'):
        os.mkdir('./log/debug_dataset/');
    print('go over')
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure();
    for i, d in enumerate(train_load,0):
        img = d[0].cpu().numpy();
        box2d_src = d[1].cpu().numpy();
        box3d_src = d[2].cpu().numpy();
        box2d_tgt = d[3].cpu().numpy();
        box3d_tgt = d[4].cpu().numpy();
        r = d[5].cpu().numpy()[0,...];
        gt = d[6].cpu().numpy()[0,...];
        gt *= np.pi;
        gt[:,1] *= 2;
        coord = np.concatenate([r.reshape(-1,1),gt.reshape(-1,2)],axis=1);
        c3dir = sph2car(coord.reshape(1,-1));
        c3d = np.zeros([2,3],dtype=np.float32);
        c3d[1,:] = mv(np.mean(box3d_src[0,:,:3],axis=0,keepdims=True))[:,:3];
        c3d[0,:] = c3d[1,:] + c3dir;
        c3d = mv_inv(c3d);
        c2d = proj(mv(c3d));
        ax = fig.add_subplot(121);
        ax.imshow(img[0,...]);
        ax.set_aspect('equal');
        ax.scatter(box2d_src[0,0:4,0],box2d_src[0,0:4,1],color='b',marker='*');
        ax.scatter(box2d_src[0,4:8,0],box2d_src[0,4:8,1],color='c',marker='*');
        ax.scatter(box2d_tgt[0,0:4,0],box2d_tgt[0,0:4,1],color='k',marker='x');
        ax.scatter(box2d_tgt[0,4:8,0],box2d_tgt[0,4:8,1],color='r',marker='x');
        ax.scatter(c2d[0,0],c2d[0,1],color='b',marker='o');
        ax.scatter(c2d[1,0],c2d[1,1],color='r',marker='o');
        ax = fig.add_subplot(122,projection='3d');
        ax.set_aspect('equal');
        ax.scatter(box3d_src[0,0:4,0],box3d_src[0,0:4,1],box3d_src[0,0:4,2],color='b',marker='*');
        ax.scatter(box3d_src[0,4:8,0],box3d_src[0,4:8,1],box3d_src[0,4:8,2],color='c',marker='*');
        ax.scatter(box3d_tgt[0,0:4,0],box3d_tgt[0,0:4,1],box3d_tgt[0,0:4,2],color='k',marker='x');
        ax.scatter(box3d_tgt[0,4:8,0],box3d_tgt[0,4:8,1],box3d_tgt[0,4:8,2],color='r',marker='x');
        ax.plot(c3d[:,0],c3d[:,1],c3d[:,2]);
        cs = np.mean(box3d_src[0,:,:3],axis=0);
        ax.scatter(cs[0],cs[1],cs[2],color='r',marker='o');
        ct = np.mean(box3d_tgt[0,:,:3],axis=0);
        ax.scatter(ct[0],ct[1],ct[2],color='b',marker='o');
        break;
    plt.show();