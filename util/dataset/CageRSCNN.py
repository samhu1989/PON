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
import h5py;
from ..tools import msk2img,write_pts2sphere;
from ..tools import draw_bounding_box_on_image as addbbx2d
from scipy.spatial.transform import Rotation as R;
import matplotlib;
matplotlib.use('agg');
import matplotlib.pyplot as plt;
#

zup = np.array([[1, 0.0000,  0.0000,  0.0000],
                [0.0000, 0,  -1.0,  0.0000],
                [0.0000, 1.0, 0, 0],
                [0.0000, 0.0000, 0.0000,  1]]
                );
                
mv = np.array([[-1.0000, -0.0000,  0.0000, -0.0000],
          [-0.0000, -0.5145,  0.8575,  0.0000],
          [0.0000,  0.8575,  0.5145, -2.4490],
          [-0.0000,  0.0000, -0.0000,  1.0000]]);
       
proj = np.array([[3.1716, 0.0000,  0.0000,  0.0000],
                [0.0000, 3.1716,  0.0000,  0.0000],
                [0.0000, 0.0000, -1.0020, -0.2002],
                [0.0000, 0.0000, -1.0000,  0.0000]]
                );

class Data(data.Dataset):
    def __init__(self, opt, train='train'):
        self.cagezip = zpf.ZipFile(opt['data_path'],'r');
        self.max_part_num = opt['max_part_num'];
        self.visible_rate = opt['visible_rate'];
        self.base_size = opt['base_size'];
        self.data = [];
        for name in self.cagezip.namelist():
            if name.endswith('.h5') and ( train in name):
                self.data.append(name);
                
    def __load__(self,h5f,fn):
        #print(fn);
        id = os.path.basename(fn);
        #print(id);
        cat = os.path.basename(os.path.dirname(fn));
        #print(cat);
        img = np.array(h5f['img']).copy().astype(np.float32);
        img = img.transpose(2,0,1);
        #print(img.min());
        #print(img.max());
        img = torch.from_numpy(img/255.0);
        pts = np.array(h5f['pts']).astype(np.float32);
        pts = torch.from_numpy(pts);
        lbl = np.array(h5f['lbl']).astype(np.int32);
        return img,pts,lbl,id,cat;
        
    def __getitem__(self,idx):
        idx = idx % self.__len__();
        fn = self.data[idx];
        h5f = h5py.File(self.cagezip.open(fn,'r'),'r');
        data = self.__load__(h5f,fn);
        #print(len(data));
        if data is None:
            return self.__getitem__(idx+1);
        else:
            return data;

    def __len__(self):
        return len(self.data);
        
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    opt['max_part_num'] = 20;
    opt['visible_rate'] = 0.5;
    opt['base_size'] = 16;
    train_data = Data(opt,'train');
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
    train_data = Data(opt,'test');
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
    r = R.from_euler('x', 90, degrees=True);
    print(r.as_dcm());
    for i, d in enumerate(train_load,0):
        img = d[0].data.cpu().numpy();
        img = img.transpose(0,2,3,1);
        ygt = d[1].data.cpu().numpy();
        cat = d[-1];
        id = d[-2];
        for ib in range(opt['batch_size']):
            opath = os.path.join('./log','rsrnn',cat[ib],id[ib]);
            if not os.path.exists(opath):
                os.makedirs(opath);
            im = Image.fromarray((img[ib,:,:,:]*255.0).astype(np.uint8))
            im.save(os.path.join(opath,'im.png'));
            pts = np.concatenate([ygt[ib,:,:],np.ones([10000,1])],axis=-1);
            ppts = np.matmul(proj,np.matmul(mv,np.matmul(zup,pts.transpose(1,0))));
            ppts = ppts.transpose(1,0);
            imp = np.ones([56,56],dtype=np.float32);
            xs = np.round(28*(ppts[:,0]/ppts[:,3])+28).astype(np.int32).flatten();
            ys = np.round(28*(-ppts[:,1]/ppts[:,3])+28).astype(np.int32).flatten();
            imp[ys.tolist(),xs.tolist()] = 0.0;
            print(id[ib]);
            print(int(id[ib].rstrip('.h5').split('_r')[1]));
            Image.fromarray((imp*255.0).astype(np.uint8),'L').save(os.path.join(opath,'proj.png'));
            write_pts2sphere(opath+'/pts.ply',ygt[ib,:,:]);
            
        break;
        
        
