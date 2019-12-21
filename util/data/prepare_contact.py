import numpy as np;
import h5py;
from PIL import Image; 

def run(**kwargs):
    root = kwargs['data_path'];
    datapath = [];
    for root, dirs, files in os.walk(self.root, topdown=True):
        for fname in files:
            if fname.endswith('.h5'):
                datapath.append(os.path.join(root,fname));
                append(os.path.basename(root));
    datapath.sort();
    for idx,p in enumerate(datapath):
        f = h5py.File(p,'r');
        cnt = f['cnt'];
        snum = cnt.shape[0];
        pnum = int(np.sum(cnt));
        #print(pnum,f['msk'].shape[0]);
        #poff = 0;
        for si in range(snum):
            img = h5file['img'][si,...];
            np.array(img)
            if si == 0:
                start = 1;
            else:
                start = 1+np.sum(cnt[:si,0]);
            end = start + cnt[si,0];
            msk = h5file['msk'][start:end,...];
            pts = h5file['pts'][start:end,...];
            p_per_s = int(cnt[si,0]);
        f.close();
