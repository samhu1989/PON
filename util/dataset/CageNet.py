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
from scipy.special import comb, perm
#
class Data(data.Dataset):
    def __init__(self, opt, train=True):
        if train:
            self.root = os.path.join(opt['data_path'],'train');
        else:
            self.root = os.path.join(opt['data_path'],'test');
        cat_lst = os.listdir(self.root);
        self.train = train;
        self.index_map = [];
        self.imap = [];
        self.jmap = [];
        self.img = [];
        self.msk = [];
        self.touch = [];
        self.box = [];
        self.cat = [];
        self.end = [];
        self.len = 0;
        cats = None;
        if 'category' in opt.keys():
            cats = opt['category'];
        for c in cat_lst:
            path = os.path.join(self.root,c)
            if os.path.isdir(path):
                f_lst = os.listdir(path);
                for f in f_lst:
                    if f.endswith('.h5'):
                        h5f = h5py.File(os.path.join(path,f),'r');
                        self.img.append(np.array(h5f['img']));
                        self.msk.append(np.array(h5f['msk']));
                        self.touch.append(np.array(h5f['touch']));
                        self.box.append(np.array(h5f['box']));
                        self.cat.append(c);
                        num = self.box[-1].shape[0];
                        pairnum = int(comb(num,2));
                        self.index_map.extend([len(self.img)-1 for x in range(pairnum)]);
                        for i in range(num-1):
                            for j in range(i+1,num):
                                self.imap.append(i);
                                self.jmap.append(j);
                                
                        if len(self.end) == 0:
                            self.end.append(pairnum);
                        else:
                            self.end.append(self.end[-1]+pairnum);
                        h5f.close();

    def __getitem__(self, idx):
        index = self.index_map[idx];
        subi = self.imap[idx];
        subj = self.jmap[idx];
        img = self.img[index];
        msk = self.msk[index];
        touch = self.touch[index];
        box = self.box[index];
        endi = self.end[index];
        msks = msk[subi,...];
        boxs = box[subi,...];
        mskt = msk[subj,...];
        boxt = box[subj,...];
        y = 0.0 ;
        for xi in range(touch.shape[0]):
            if subi == touch[xi,0] and subj == touch[xi,1]:
                y = 1.0;
            if subj == touch[xi,0] and subi == touch[xi,1]:
                y = 1.0;
        img = torch.from_numpy(img)
        msks = torch.from_numpy(msks)
        mskt = torch.from_numpy(mskt)
        y = torch.from_numpy(np.array([y],dtype=np.float32))
        vec = np.zeros([21],dtype=np.float32);
        #
        vec[:3] = boxs[:3];
        vec[3:9] = boxs[6:12];
        #
        vec[9:12] = boxt[:3];
        vec[12:15] = boxt[3:6] - boxs[3:6];
        vec[15:21] = boxt[6:12];
        #
        vec = torch.from_numpy(vec);
        return img,msks,mskt,y,vec,self.cat[index];

    def __len__(self):
        return len(self.index_map);
        
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
