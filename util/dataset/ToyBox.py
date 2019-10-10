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
        img = np.array(Image.open(imgname)).astype(np.float32);
        img /= 255.0;
        return torch.from_numpy(img.copy()),torch.from_numpy(pts.copy());

    def __len__(self):
        return len(self.datapath);
        
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
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure();
    for i, d in enumerate(train_load,0):
        img = d[0].cpu().numpy();
        pts = d[1].cpu().numpy();
        ax = fig.add_subplot(1,2,1);
        ax.imshow(img[0,:,:,0:3]);
        ax = fig.add_subplot(1,2,2,projection='3d');
        ax.plot(pts[0,:,0],pts[0,:,1],pts[0,:,2],'r',marker='x');
        break;
    plt.show();
    
        
