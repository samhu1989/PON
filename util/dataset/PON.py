from __future__ import print_function
from PIL import Image
#sys import
import os;
import random;
import numpy as np;
#torch import
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
#project import
from ..data.ply import read_ply;
#
class Data(data.Dataset):
    def __init__(self, opt, train=True):
        self.SVR = (opt['mode']=='SVR');
        self.pts_num = opt['pts_num_gt'];
        self.root = opt['data_path'];
        self.train = train
        self.datapath = [];
        self.transforms = transforms.Compose([
                             transforms.Resize(size =  224, interpolation = Image.BILINEAR),
                             transforms.ToTensor(),
                             # normalize,
                        ])

        # RandomResizedCrop or RandomCrop
        self.randcrop = transforms.Compose([
                                         transforms.RandomCrop(214),
                                         transforms.RandomHorizontalFlip(),
                            ])
        self.centercrop = transforms.Compose([
                        transforms.CenterCrop(214),
                        ])
        if self.train:
            fpath = self.root+os.sep+'train';
        else:
            fpath = self.root+os.sep+'val';
        cats = opt['category'];
        f_lst = os.listdir(fpath);
        for f in f_lst:
            if f.endswith('_pts.ply') and ( ( f.split('_')[0] in cats ) or (not cats) ):
                self.datapath.append(fpath+os.sep+f.rstrip('_pts.ply'));

    def __getitem__(self, index):
        fn = self.datapath[index];
        data = {}
        ply_data = read_ply(fn+'_pts.ply');
        points = ply_data['points'];
        pts = np.array(points)[:,:3];
        row_rand_array = np.arange(pts.shape[0])
        np.random.shuffle(row_rand_array)
        row_rand_pts = pts[row_rand_array[0:self.pts_num]];
        pts = torch.from_numpy(row_rand_pts).contiguous();
        cat = os.path.basename(fn).split('.')[0];
        cat = fn.split('_')[0];
        # load image
        if self.SVR:
            v_path = fn+'_view';
            v = os.listdir(v_path);
            v_lst = [];
            for f in v:
                if not f.endswith('.png0001.png'):
                    v_lst.append(f);
            random.shuffle(v_lst)
            im = Image.open(os.path.join(v_path,v_lst[0]));
            if self.train:
                im = self.randcrop(im) #random crop
            else:
                im = self.centercrop(im) #center crop
            im = self.transforms(im) #scale
            im = im[:3,:,:]
        else:
            im = 0;
        return im,pts,cat;

    def __len__(self):
        return len(self.datapath)
