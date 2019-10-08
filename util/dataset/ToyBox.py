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
#project import
#
class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.root = opt['data_path'];
        self.train = train
        self.datapath = [];
        
                
    def __getitem__(self, index):
        return self.load(index);

    def getpartimg(self,img,msk):
        msk = ndimage.grey_dilation(msk,size=(10,10));
        th = 150.0;
        if not self.train:
            th = 100.0
        if np.sum(msk) < th:
            return None;
        else:
            return img*msk.reshape(msk.shape[0],msk.shape[1],1);
        
    def load(self,idx):
        partimg = None;
        try:
            index = idx%self.__len__();
            findex = self.fmap[index];
            h5file = h5py.File(self.datapath[findex],'r');            
            img = h5file['img'][self.smap[index],...];
            msk = h5file['msk'][self.pmap[index],...];
            pts = h5file['pts'][self.pmap[index],...];
            partimg = self.getpartimg(img,msk);
            h5file.close();
        except Exception as e:
            print(e);
            traceback.print_exc();
            exit();
        if not partimg is None:
            im = partimg.copy();
            im = im.transpose(2,0,1)
            pts = pts.copy();
            pts = pts - np.mean(pts,axis=0,keepdims=True);
            return torch.from_numpy(im),torch.from_numpy(pts),self.cat[self.fmap[index]];
        else:
            return self.load(index+1);

    def __len__(self):
        return len(self.pmap);
        
#debuging the dataset      
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    train_data = Data(opt,True);
    val_data = Data(opt,False);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    val_load = data.DataLoader(val_data,batch_size=1,shuffle=False,num_workers=opt['workers']);
    if not os.path.exists('./log/debug_dataset/'):
        os.mkdir('./log/debug_dataset/');
    print('go over')
    for i, d in enumerate(train_load,0):
        partimg = d[0].cpu().numpy();
        partpts = d[1].cpu().numpy();
        print(i,'/',len(train_data) // opt['batch_size']);
        
