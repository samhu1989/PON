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
from PIL import Image;
import zipfile as zpf;
#
class Data(data.Dataset):
    def __init__(self, opt, train='train'):
        self.cagezip = zpf.ZipFile(opt['data_path'],'r');
        self.data = [];
        for name in self.cagezip.namelist():
            if name.endswith('.h5') and ( train in name):
                self.data.append(name);
        

    def __load__(self,path):
        h5f = h5py.File(path,'r');
        cageh5 = h5py.File(cagef,'r');
        imtmp = Image.fromarray(np.array(h5f['img448']));
        imtmp = np.array(imtmp.resize(size=[224,224])).astype(np.float32)/255.0;
        out = {
            'img':imtmp,
            'msk':np.array(h5f['msk']),
            'smsk':np.array(h5f['smsk']),
            'touch':np.array(h5f['touch']),
            'box':np.array(h5f['box'])
        };
        h5f.close();
        return out;
        
    def __getitem__(self, idx):
        idx = idx % self.__len__();
        index = self.index_map[idx];
        subi = self.imap[idx];
        subj = self.jmap[idx];
        data = self.__load__(self.files[index]);

        return img, im_info, gt_boxes, num_boxes;

    def __len__(self):
        return len(self.data);
        
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    train_data = Data(opt,'train');
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    for i, d in enumerate(train_load,0):
        print(d);
