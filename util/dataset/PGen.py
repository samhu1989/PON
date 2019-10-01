from __future__ import print_function
from PIL import Image
#sys import
import os;
import random;
import h5py;
import numpy as np;
from scipy import ndimage;
#torch import
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from ..data.ply import write_ply;
import pandas as pd;
#project import
#
class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.root = opt['data_path'];
        self.train = train
        self.datapath = [];
        self.datafile = []; 
        fs = os.listdir(self.root);
        self.cnt = 0;
        self.fmap = [];
        self.smap = [];
        self.pmap = [];
        fs.sort();
        for f in fs:
            if f.endswith('.h5'):
                if self.train and ('train' in f):
                    self.datapath.append(os.path.join(self.root,f));
                elif not self.train and ('test' in f):
                    self.datapath.append(os.path.join(self.root,f));
        print(self.datapath[0]);
        for idx,p in enumerate(self.datapath):
            self.datafile.append(h5py.File(p,'r'));
            cnt = self.datafile[-1]['cnt'];
            snum = cnt.shape[0];
            pnum = int(np.sum(cnt));
            self.cnt += pnum;
            poff = 0;
            for si in range(snum):
                p_per_s = int(cnt[si,0]);
                self.fmap.extend([idx]*p_per_s);
                self.smap.extend([si]*p_per_s);
            self.pmap.extend([x for x in range(1,pnum)]);
                
    def __getitem__(self, index):
        return self.load(index);

    def getpartimg(self,img,msk):
        msk = ndimage.grey_dilation(msk,size=(10,10))
        if np.sum(msk) < 200.0:
            return None;
        else:
            return img*msk.reshape(msk.shape[0],msk.shape[1],1);
        
    def load(self,index):
        h5file = self.datafile[self.fmap[index]];
        img = h5file['img'][self.smap[index],...];
        msk = h5file['msk'][self.pmap[index],...];
        pts = h5file['pts'][self.pmap[index],...];
        partimg = self.getpartimg(img,msk);
        if not partimg is None:
            return torch.from_numpy(partimg.copy()),torch.from_numpy(pts.copy());
        else:
            return self.load(index+1);

    def __len__(self):
        return self.cnt;
        
#debuging the dataset      
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    train_data = Data(opt,True);
    val_data = Data(opt,False);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    val_load = data.DataLoader(val_data,batch_size=1,shuffle=False,num_workers=opt['workers']);
    for i, d in enumerate(train_load,0):
        partimg = d[0].cpu().numpy();
        partpts = d[1].cpu().numpy();
        for j in range(partimg.shape[0]):
            im = Image.fromarray((partimg[j,...]*255.0).astype(np.uint8),'RGB');
            im.save('./log/debug_dataset/im_%03d_%03d.png'%(i,j));
            write_ply('./log/debug_dataset/pt_%03d_%03d.ply'%(i,j),points=pd.DataFrame(partpts[j,...]));
        if i > 1:
            break;
        
