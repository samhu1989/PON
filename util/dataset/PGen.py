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
from ..data.ply import write_ply;
import pandas as pd;
import traceback;
#project import
#
class Data(data.Dataset):
    def __init__(self,opt,train=True):
        self.root = opt['data_path'];
        self.train = train
        self.datapath = [];
        self.datadict = [];
        self.cnt = 0;
        self.fmap = [];
        self.smap = [];
        self.pmap = [];
        self.cat = [];
        self.load_length = 3;
        self.load_dict = {};
        for root, dirs, files in os.walk(self.root, topdown=True):
            for fname in files:
                if fname.endswith('.h5'):
                    if self.train and ('train' in fname):
                        self.datapath.append(os.path.join(root,fname));
                    elif not self.train and ('test' in fname):
                        self.datapath.append(os.path.join(root,fname));
                    self.cat.append(os.path.basename(root));
        self.datapath.sort();
        for idx,p in enumerate(self.datapath):
            print(p);
            f = h5py.File(p,'r');
            cnt = f['cnt'];
            snum = cnt.shape[0];
            pnum = int(np.sum(cnt));
            self.cnt += pnum;
            poff = 0;
            for si in range(snum):
                p_per_s = int(cnt[si,0]);
                self.fmap.extend([idx]*p_per_s);
                self.smap.extend([si]*p_per_s);
            self.pmap.extend([x for x in range(1,pnum)]);
            f.close();
        
                
    def __getitem__(self, index):
        return self.load(index);

    def getpartimg(self,img,msk):
        msk = ndimage.grey_dilation(msk,size=(10,10))
        if np.sum(msk) < 200.0:
            return None;
        else:
            return img*msk.reshape(msk.shape[0],msk.shape[1],1);
        
    def load(self,index):
        try:
            findex = self.fmap[index];
            if findex in self.load_dict.keys():
                h5file = self.load_dict[findex];
            else:
                h5file = h5py.File(self.datapath[findex],'r');
                self.load_dict[findex] = {};
                for k in h5file.keys():
                    self.load_dict[findex][k] = np.array(h5file[k]).copy();
                h5file.close();
                h5file = self.load_dict[findex];
            img = h5file['img'][self.smap[index],...];
            msk = h5file['msk'][self.pmap[index],...];
            pts = h5file['pts'][self.pmap[index],...];
            partimg = self.getpartimg(img,msk);
            if len(self.load_dict) > self.load_length:
                self.load_dict.pop(self.load_dict.keys()[0]);
        except Exception as e:
            print(e);
            traceback.print_exc();
            exit();
        if not partimg is None:
            im = partimg.copy();
            im = im.transpose(2,0,1)
            return torch.from_numpy(im),torch.from_numpy(pts.copy()),self.cat[self.fmap[index]];
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
    if not os.path.exists('./log/debug_dataset/'):
        os.mkdir('./log/debug_dataset/');
    for i, d in enumerate(train_load,0):
        partimg = d[0].cpu().numpy();
        partpts = d[1].cpu().numpy();
        for j in range(partimg.shape[0]):
            im = Image.fromarray((partimg[j,...]*255.0).astype(np.uint8),'RGB');
            im.save('./log/debug_dataset/%s_im_%03d_%03d.png'%(d[-1][j],i,j));
            write_ply('./log/debug_dataset/%s_pt_%03d_%03d.ply'%(d[-1][j],i,j),points=pd.DataFrame(partpts[j,...]));
        if i > 1:
            break;
        
