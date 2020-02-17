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
#
class Data(data.Dataset):
    def __init__(self, opt, train='train'):
        if train:
            self.root = os.path.join(opt['data_path'],train);
        else:
            self.root = os.path.join(opt['data_path'],train);
        self.rate = opt['user_rate'];
        cat_lst = os.listdir(self.root);
        self.train = train;
        self.index_map = [];
        self.imap = [];
        self.jmap = [];
        self.img = [];
        self.msk = [];
        self.smsk = [];
        self.touch = [];
        self.box = [];
        self.cat = [];
        self.id = [];
        self.end = [];
        self.len = 0;
        cats = None;
        if 'category' in opt.keys():
            cats = opt['category'];
        for c in cat_lst:
            path = os.path.join(self.root,c);
            print('loading:',c);
            if os.path.isdir(path):
                f_lst = os.listdir(path);
                for fidx,f in enumerate(f_lst):
                    if f.endswith('.h5'):
                        print(fidx);
                        h5f = h5py.File(os.path.join(path,f),'r');
                        imtmp = Image.fromarray(np.array(h5f['img448']));
                        imtmp = np.array(imtmp.resize(224,224)).astype(np.float32)/255.0;
                        self.img.append(imtmp);
                        self.msk.append(np.array(h5f['msk']));
                        self.smsk.append(np.array(h5f['smsk']));
                        self.touch.append(np.array(h5f['touch']));
                        self.box.append(np.array(h5f['box']));
                        self.cat.append(c);
                        self.id.append(os.path.basename(f).split('.')[0]);
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
        idx = idx % self.__len__();
        index = self.index_map[idx];
        subi = self.imap[idx];
        subj = self.jmap[idx];
        img = self.img[index];
        msk = self.msk[index];
        smsk = self.smsk[index];
        touch = self.touch[index];
        box = self.box[index];
        endi = self.end[index];
        msks = msk[subi,...];
        smsks = smsk[subi,...];
        boxs = box[subi,...];
        mskt = msk[subj,...];
        smskt = smsk[subj,...];
        boxt = box[subj,...];
        if (( np.sum(msks) / np.sum(smsks) ) < self.rate) or (( np.sum(mskt) / np.sum(smskt) ) < self.rate ):
            return self.__getitem__(idx+1);
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
        boxs = torch.from_numpy(boxs.astype(np.float32));
        boxt = torch.from_numpy(boxt.astype(np.float32));
        return img,msks,mskt,y,vec,boxs,boxt,self.id[index],self.cat[index];

    def __len__(self):
        return len(self.index_map);
        
def rot(r1,r2):
    rr1 = r1 / np.linalg.norm(r1);
    rr2 = r2 - np.dot(r2,rr1)*rr1;
    rr2 = rr2 / np.linalg.norm(rr2);
    r3 = np.cross(rr1,rr2);
    rot = np.stack([rr1,rr2,r3],axis=0);
    return rot;
        
def parse(vec):
    coord = np.array([[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]],dtype=np.float32);
    print(vec);
    ss = vec[:3];
    coords = ss[np.newaxis,:]*coord;
    sr1 = vec[3:6];
    sr2 = vec[6:9]
    srot = rot(sr1,sr2);
    print('srot',srot)
    vs = np.dot(coords,srot.reshape(3,3))
    ts = vec[9:12];
    coordt = ts[np.newaxis,:]*coord;
    center = vec[12:15];
    tr1 = vec[15:18];
    tr2 = vec[18:21];
    trot = rot(tr1,tr2);
    print('trot',trot)
    vt = np.dot(coordt,trot.reshape(3,3)) + center[np.newaxis,:];
    return  vs,vt;
        
def run(**kwargs):
    opt = kwargs;
    opt['workers'] = 0;
    train_data = Data(opt,True);
    train_load = data.DataLoader(train_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    tri = box_face;
    fidx = tri
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    bi = 2 % opt['batch_size'];
    for i, d in enumerate(train_load,0):
        img = d[0].data.cpu().numpy()[bi,...];
        msks = d[1].data.cpu().numpy()[bi,...];
        mskt = d[2].data.cpu().numpy()[bi,...];
        vec = d[4].data.cpu().numpy()[bi,...];
        Image.fromarray((img*255).astype(np.uint8),mode='RGB').save('./log/im.png');
        Image.fromarray((msks*255).astype(np.uint8),mode='L').save('./log/mks.png');
        Image.fromarray((mskt*255).astype(np.uint8),mode='L').save('./log/mkt.png');
        Image.fromarray((img*(msks+mskt)[:,:,np.newaxis]*255).astype(np.uint8),mode='RGB').save('./log/mkd.png');
        ptsa,ptsb = parse(vec);
        write_ply('./log/pa.ply',points=pd.DataFrame(ptsa),faces=pd.DataFrame(face));
        write_ply('./log/pb.ply',points=pd.DataFrame(ptsb),faces=pd.DataFrame(face));
        ba = d[5].data.cpu().numpy()[0,...];
        bb = d[6].data.cpu().numpy()[0,...];
        print('a')
        pba = OBB.v2points(ba);
        print('b')
        pbb = OBB.v2points(bb);
        write_ply('./log/pba.ply',points=pd.DataFrame(pba),faces=pd.DataFrame(face));
        write_ply('./log/pbb.ply',points=pd.DataFrame(pbb),faces=pd.DataFrame(face));
        break;
            
        
        
        
