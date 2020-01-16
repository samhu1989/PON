import os;
import h5py;
from scipy.spatial.transform import Rotation as R;
from PIL import Image;
import numpy as np;
from .ply import read_ply,write_ply
from .obb import OBB;
from .gen_toybox import box_face as bf;
import pandas as pd;

def run(**kwargs):
    dp = './data/cagenet';
    sp = './data/cage';
    op = './data/cagenet2'
    
    for root, dirs, files in os.walk(dp, topdown=True):
        for name in files:
            if name.endswith('.h5'):
                id = name.rstrip('.h5');
                cat = os.path.basename(root);
                sub = os.path.basename(os.path.dirname(root));
                if not os.path.exists(os.path.join(op,sub,cat)):
                    os.makedirs(os.path.join(op,sub,cat));
                print(sub,cat,id);
                h5fs = h5py.File(os.path.join(sp,sub,cat,id,'pn.h5'),'r');
                h5fd = h5py.File(os.path.join(dp,sub,cat,id+'.h5'),'r');
                if os.path.exists(os.path.join(op,sub,cat,id+'.h5')):
                    continue;
                h5fo = h5py.File(os.path.join(op,sub,cat,id+'.h5'),'w');
                h5fo.create_dataset("img",data=h5fd['img'],compression="gzip", compression_opts=9);
                h5fo.create_dataset("touch",data=h5fd['touch'],compression="gzip", compression_opts=9);
                #get angle
                imgp = os.path.join(sp,sub,cat,id,'sp','models');
                imgs = os.listdir(imgp);
                angle = None;
                for fs in imgs:
                    if fs.endswith('.png') and 'model_normalized_align.obj' in fs:
                        angle = int((fs.split('_')[3]).split('.')[0]);
                        img = Image.open(os.path.join(imgp,fs));
                        img = np.array(img).astype(np.float32) / 255.0;
                partp = os.path.join(sp,sub,cat,id,'parts2');
                label = np.array(h5fs['label']);
                start = np.min(label);
                end = np.max(label);
                cnt = 0;
                msklst = [];
                smsklst = [];
                ps = [];
                r1 = R.from_euler('x',-90,degrees=True);
                r2 = R.from_euler('y',-angle,degrees=True);
                for i in range(start,end+1):
                    num = np.sum(label == i);
                    if num > 0 :
                        mskp = 'p_%d_b_%d_msk0001.png'%(cnt,angle);
                        mskpb = 'p_%d_b_%d_mskb0001.png'%(cnt,angle);
                        msk = Image.open(os.path.join(partp,'all_msk',mskp));
                        msk = np.array(msk).astype(np.float32) / 255.0;
                        msk = msk[:,:,2];
                        smsk = Image.open(os.path.join(partp,'self_msk',mskp));
                        smsk = np.array(smsk).astype(np.float32) / 255.0;
                        smsk = smsk[:,:,2];
                        if np.sum(msk) > 9:
                            cpath = os.path.join(partp,'p_%d_b.ply'%cnt);
                            pts = read_ply(cpath);
                            pv = np.array(pts['points']);
                            fvidx = np.unique(np.array(pts['mesh']).flatten());
                            pv = pv[fvidx,:3];
                            pv = r1.apply(pv);
                            pv = r2.apply(pv);
                            ps.append(np.array(pv));
                            m = Image.fromarray(np.array(msk*255.0).astype(np.uint8),mode='L');
                            m.save(os.path.join(partp,'all_msk',mskpb));
                            msklst.append(msk);
                            sm = Image.fromarray(np.array(smsk*255.0).astype(np.uint8),mode='L');
                            sm.save(os.path.join(partp,'self_msk',mskpb));
                            smsklst.append(smsk);
                        cnt += 1;
                #print(len(ps));
                obblst = [];
                obbp = [];
                obbf = [];
                obbcnt = 0;
                for pts in ps:
                    obba = OBB.build_by_trimesh(pts);
                    obbb = OBB.build_from_points(pts);
                    if (obba is None) or (obba.volume > obbb.volume):
                        obbr = obbb;
                    else:
                        obbr = obba;
                    #print('size:',obba.volume);
                    obblst.append( obbr );
                    obbf.append(bf + obbcnt*8);
                    obbcnt += 1;
                #print('obbf:',len(obbf));
                #print('obbcnt:',obbcnt);
                msks = np.stack(msklst,axis=0);
                h5fo.create_dataset("msk", data=msks,compression="gzip", compression_opts=9);
                smsks = np.stack(smsklst,axis=0);
                h5fo.create_dataset("smsk", data=smsks,compression="gzip", compression_opts=9);
                obbk = [];
                for obb in obblst:
                    obbp.append(obb.points);
                    obbk.append(obb.tov);
                obbv = np.concatenate(obbp,axis=0);
                h5fo.create_dataset("box",data=np.stack(obbk,axis=0),compression="gzip", compression_opts=9);
                fidx = np.concatenate(obbf,axis=0);
                T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
                face = np.zeros(shape=[12*len(obbf)],dtype=T);
                for i in range(fidx.shape[0]):
                    face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
                obox = os.path.join(op,sub,cat,id+'_box.ply');
                write_ply(obox,points=pd.DataFrame(obbv.astype(np.float32)),faces=pd.DataFrame(face));  
                #exit();
