from __future__ import print_function
from PIL import Image
#sys import
import os;
import random;
import numpy as np;
import h5py;
#torch import
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
#project import
from ..data.obb import OBB;
from ..data.gen_toybox import box_face;
import pandas as pd;
from ..data.ply import write_ply;
#
class Data(data.Dataset):
    def __init__(self, opt, train=True):
        self.root = opt['data_path'];
        self.train = train;
        f_lst = os.listdir(self.root);
        self.data = [];
        self.cat = [];
        self.len = 0;
        cats = None;
        if 'category' in opt.keys():
            cats = opt['category'];
        for f in f_lst:
            if f.endswith('.h5'):
                h5f = h5py.File( os.path.join(self.root,f) , 'r' );
                X = np.array(h5f['box_pair']);
                catname = os.path.basename(f).split('.')[0];
                if ( cats is not None ) and (not (catname in cats)):
                    continue;
                for i in range(X.shape[0]):
                    self.data.append(X[i,:].astype(np.float32));
                    self.cat.append(catname);
                self.len += X.shape[0];

    def __getitem__(self, idx):
        index = idx % self.len;
        return torch.from_numpy(self.data[index]),self.cat[index];

    def __len__(self):
        return self.len;
        
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    train_data = Data(opt,True);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    tri = box_face;
    fidx = tri
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    for i, d in enumerate(train_load,0):
        fig = plt.figure(figsize=(32,64));
        X = d[0];
        cat = d[1];
        x = X.cpu().numpy()[0,...];
        obba = OBB();
        ca = x[:3];
        ea = x[3:6];
        ra = x[6:12];
        ra3 = np.cross(x[6:9],x[9:12]); 
        r = np.zeros((3,3),dtype=np.float32);
        r[0,:] = x[6:9];
        r[1,:] = x[9:12];
        r[2,:] = ra3;
        obba.rotation = r;
        centroid = np.dot(ca,np.linalg.inv(r));
        obba.min = centroid - ea;
        obba.max = centroid + ea;
        ptsa = np.stack(obba.points,axis=0).astype(np.float32);
        #=================
        obbb = OBB();
        cb = x[12:15];
        eb = x[15:18];
        rb = x[18:24];
        rb3 = np.cross(x[18:21],x[21:24]); 
        r = np.zeros((3,3),dtype=np.float32);
        r[0,:] = x[18:21];
        r[1,:] = x[21:24];
        r[2,:] = rb3;
        obbb.rotation = r;
        centroid = np.dot(cb,np.linalg.inv(r));
        obbb.min = centroid - eb;
        obbb.max = centroid + eb;
        ptsb = np.stack(obbb.points,axis=0).astype(np.float32);
        #
        ax = fig.add_subplot(1,2,1,projection='3d');
        ax.view_init(elev=20, azim=90)
        ax.set_aspect('equal', adjustable='box');
        ax.set_xlim([-1,1]);
        ax.set_ylim([-1,1]);
        ax.set_zlim([-1,1]); 
        #
        ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
        ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
        #
        ax = fig.add_subplot(1,2,2,projection='3d');
        ax.set_aspect('equal', adjustable='box');
        ax.set_xlim([-1,1]);
        ax.set_ylim([-1,1]);
        ax.set_zlim([-1,1]);
        #
        ax.plot_trisurf(ptsa[...,0],ptsa[...,2],tri,ptsa[...,1],color=(0,0,1,0.1));
        ax.plot_trisurf(ptsb[...,0],ptsb[...,2],tri,ptsb[...,1],color=(0,1,0,0.1));
        #
        write_ply('./log/debuga.ply',points=pd.DataFrame(ptsa),faces=pd.DataFrame(face));
        write_ply('./log/debugb.ply',points=pd.DataFrame(ptsb),faces=pd.DataFrame(face));
        print(cat[0])
        #
        plt.show();
        plt.close(fig);
    return;
