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
#project import
#
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
            if (not 'msk' in f) and f.endswith('.ply'):
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
        plyname = self.datapath[index];
        imgname = plyname.replace('.ply','.ply_r_000.png');
        data = read_ply(plyname);
        pts = np.array(data['points']);
        fidx = np.array(data['mesh']);
        num = self.pts_num // fidx.shape[0];
        img = np.array(Image.open(imgname)).astype(np.float32);
        img /= 255.0;
        imgtensor = torch.from_numpy(img.copy().transpose(2,0,1)).contiguous();
        #pick msks for target and source boxes
        idx = 0;
        mskname = plyname.replace('.ply','_msk%02d.ply_r_000_albedo.png0001.png'%idx);
        msks = [];
        while os.path.exists(mskname):
            msks.append(mskname);
            idx += 1;
            mskname = plyname.replace('.ply','_msk%02d.ply_r_000_albedo.png0001.png'%idx);
        msknum = len(msks);
        #
        pick = np.random.randint(0,msknum);
        if pick == 0:
            srcpick = np.random.randint(1,len(msks));
        else:
            if len(msks) > 2:
                alpha = np.random.randint(0,2)
                srcpick = alpha*np.random.randint(0,bspick)+(1-alpha)*np.random.randint(bspick+1,msknum);
            else:
                srcpick = 0;
        #
        boxall = torch.from_numpy(pts.copy());
        boxpts = tri2pts(boxall,fidx,num).transpose(1,0).contiguous();
        #
        msk = np.array(Image.open(msks[pick])).astype(np.float32);
        msk = msk.copy();
        msk = msk[:,:,2] / 255.0;
        tgt_msk = torch.from_numpy(msk).contiguous();
        tgt_box = boxall[8*pick:8*(pick+1),:].contiguous();
        #
        msk = np.array(Image.open(msks[srcpick])).astype(np.float32);
        msk = msk.copy();
        msk = msk[:,:,2] / 255.0;
        src_msk = torch.from_numpy(msk).contiguous();
        src_box = boxall[8*srcpick:8*(srcpick+1),:].contiguous();
        all_box = boxall.contiguous();
        return imgtensor,src_msk,tgt_msk,src_box,tgt_box,all_box,boxpts,'box';

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
        src_msk = d[1].cpu().numpy();
        tgt_msk = d[2].cpu().numpy();
        src_box = d[3].cpu().numpy();
        tgt_box = d[4].cpu().numpy();
        all_box = d[5].cpu().numpy();
        ax = fig.add_subplot(3,2,1);
        ax.imshow(img[0,0:3,:,:].transpose(1,2,0));
        ax = fig.add_subplot(3,2,2);
        ax.imshow(src_msk[0,:,:],cmap='gray');
        ax = fig.add_subplot(3,2,3);
        ax.imshow(tgt_msk[0,:,:],cmap='gray');
        coord = torch.from_numpy((np.mgrid[-1:1:224j,-1:1:224j]).astype(np.float32));
        coord = coord.type(d[0].type());
        ax = fig.add_subplot(3,2,4);
        coordim = coord.cpu().numpy();
        coordim = np.concatenate([coordim,np.zeros([1,224,224],dtype=np.float32)],axis=0);
        coordim += 1.0;
        coordim *= 0.5;
        ax.imshow(coordim.transpose(1,2,0));
        ax = fig.add_subplot(3,2,6,projection='3d');
        ax.scatter(src_box[0,:,0],src_box[0,:,1],src_box[0,:,2],marker='x');
        ax.scatter(tgt_box[0,:,0],tgt_box[0,:,1],tgt_box[0,:,2],marker='o');
        ax.scatter(all_box[0,:,0],all_box[0,:,1],all_box[0,:,2],marker='^');
        break;
    plt.show();
    
        
