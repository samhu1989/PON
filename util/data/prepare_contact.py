import numpy as np;
import h5py;
from PIL import Image;
import os;
from .obb import OBB;
from .gen_toybox import box_face as bf ;
from .ply import write_ply
import pandas as pd;
import scipy;

def run(**kwargs):
    root = kwargs['data_path'];
    datapath = [];
    for root, dirs, files in os.walk(root, topdown=True):
        for fname in files:
            if fname.endswith('.h5') and 'train' in fname:
                datapath.append(os.path.join(root,fname));
    trainnum = 1000;
    testnum = 100;
    for idx,p in enumerate(datapath):
        f = h5py.File(p,'r');
        cnt = f['cnt'];
        snum = cnt.shape[0];
        pnum = int(np.sum(cnt));
        for si in range(1,snum+1):
            if int(cnt[si,0]) > 20:
                continue;
            if 'train' in p:
                print('train:',trainnum)
                if trainnum <= 0:
                    exit();
                path = './data/cagenet/train/Chair/%05d.h5'%trainnum;
                trainnum -= 1;
                if os.path.exists(path):
                    continue;
                outf = h5py.File(path,'w');
            elif 'test' in p:
                print('test:',testnum)
                if testnum <= 0:
                    exit();
                path = './data/cagenet/test/Chair/%05d.h5'%testnum
                testnum -= 1;
                if os.path.exists(path):
                    continue;
                outf = h5py.File(path,'w');

            img = f['img'][si,...];
            outf.create_dataset("img", data=np.array(img),compression="gzip", compression_opts=9)
            im = Image.fromarray(np.array(img*255.0).astype(np.uint8));
            #im.save('./log/debug/im.png');
            if si == 0:
                start = 1;
            else:
                start = int(1+np.sum(cnt[:si,0]));
            end = start + int(cnt[si,0]);
            ps = [];
            msklst = [];
            for pi in range(start,end):
                msk = f['msk'][pi,...];
                pts = f['pts'][pi,...];
                if np.sum(msk) > 25:
                    ps.append(np.array(pts));
                    m = Image.fromarray(np.array(msk*255.0).astype(np.uint8),mode='L');
                    msklst.append(np.array(msk));
                    #m.save('./log/debug/msk_%d.png'%pi);
            if len(msklst) < 2:
                outf.close();
                os.remove(path);
                trainnum += 1;
                continue;
            msks = np.stack(msklst,axis=0);
            print('msks',msks.shape);
            outf.create_dataset("msk", data=msks,compression="gzip", compression_opts=9);
            obblst = [];
            obbp = [];
            obbf = [];
            obbcnt = 0;
            for pts in ps:
                obblst.append( OBB.build_from_points(pts) );
                obbf.append(bf + obbcnt*8);
                obbcnt += 1;
            num = len(ps);
            mm = [];
            for pi in range(num-1):
                for pj in range(pi+1,num):
                    da,db = ( ps[pi],ps[pj] ) if obblst[pi].volume > obblst[pj].volume else ( ps[pj], ps[pi]);
                    tree = scipy.spatial.KDTree(da);
                    dsta,idxa = tree.query(da,k=2);
                    dstb,idxb = tree.query(db,k=1);
                    if np.min(dstb) < np.mean(dsta[:,1]):
                        mm.append(np.array([pi,pj],dtype=np.int32));
            print('mm:',len(mm));
            if len(mm) < 1:
                outf.close();
                os.remove(path);
                trainnum += 1;
                continue;
            outf.create_dataset("touch",data=np.stack(mm,axis=0),compression="gzip", compression_opts=9);
            obbk = [];
            for obb in obblst:
                obbp.append(obb.points);
                obbk.append(obb.tov);
            obbv = np.concatenate(obbp,axis=0);
            outf.create_dataset("box",data=np.stack(obbk,axis=0),compression="gzip", compression_opts=9);
            fidx = np.concatenate(obbf,axis=0);
            T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
            face = np.zeros(shape=[12*len(obbf)],dtype=T);
            for i in range(fidx.shape[0]):
                face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
            if testnum <= 0 and trainnum <= 0 :
                exit();
            #write_ply('./log/debug/box.ply',points=pd.DataFrame(obbv.astype(np.float32)),faces=pd.DataFrame(face));
        f.close();
