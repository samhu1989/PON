import h5py;
import os;
import numpy as np;

def run(**kwargs):
    data_root = kwargs['data_path'];
    res_root = kwargs['user_key'];
    num = kwargs['batch_size'];
    datapath = [];
    for root, dirs, files in os.walk(data_root, topdown=True):
            for fname in files:
                if fname.endswith('.h5'):
                    datapath.append(os.path.join(root,fname));
    datapath.sort();
    h5f = h5py.File(res_root+os.sep+'%03d_tiny.h5'%num,'w');
    img_ds = h5f.create_dataset("img",(1,224,224,3),maxshape=(None,224,224,3),chunks=(1,224,224,3),compression="gzip", compression_opts=9);
    pts_ds = h5f.create_dataset("pts",(1,1000,3),maxshape=(None,1000,3),chunks=(1,1000,3),compression="gzip", compression_opts=9);
    msk_ds = h5f.create_dataset("msk",(1,224,224),maxshape=(None,224,224),chunks=(1,224,224),compression="gzip", compression_opts=9);
    cnt_ds = h5f.create_dataset("cnt",(1,1),maxshape=(None,1),chunks=(1,1),compression="gzip", compression_opts=9);
    for idx,p in enumerate(datapath):
        f = h5py.File(p,'r');
        cnt = f['cnt'];
        img = f['img'];
        msk = f['msk'];
        pc = f['pts'];
        for i in range(cnt.shape[0]):
            if int(cnt[i,:])==0:
                continue;
            cnum = cnt[i,...];
            img_ds.resize((img_ds.shape[0] + 1), axis=0);
            img_ds[-1,...] = img[i,...].copy();
            cnt_ds.resize((cnt_ds.shape[0] + 1), axis=0);
            cnt_ds[-1,0] = 0;
            start = int(np.sum(cnt[0:i,0]))+1;
            for ic in range(int(cnum)):
                msk_ds.resize((msk_ds.shape[0] + 1), axis=0);
                msk_ds[-1,...] = msk[start+ic,...].copy();
                pts_ds.resize((pts_ds.shape[0] + 1), axis=0);
                pts_ds[-1,...] = pc[start+ic,...].copy();
                cnt_ds[-1,0] += 1;
                print(num);
                num -= 1;
                if num <= 0:
                    break;
            if num <= 0 :
                break;
        if num <= 0 :
            break;

            