import numpy as np;
import json;
from ply import read_ply, write_ply;
import h5py;
from PIL import Image;
from obb import OBB;
import OpenEXR, Imath;
import os;

def packorigin(imgp,angle,h5fo):
    img = Image.open(os.path.join(imgp,'model_normalized_r%d.png'%angle));
    img = np.array(img);
    h5fo.create_dataset("img448",data=img,compression="gzip", compression_opts=9);
    dimg = OpenEXR.InputFile(os.path.join(imgp,'model_normalized_r%d_depth.exr0001.exr'%angle));
    dr,dg,db = dimg.channels("RGB");
    dr = np.fromstring(dr,dtype=np.float32).reshape((448,448));
    dg = np.fromstring(dg,dtype=np.float32).reshape((448,448));
    db = np.fromstring(db,dtype=np.float32).reshape((448,448));
    h5fo.create_dataset("depth448",data=np.stack([dr,dg,db],axis=2),compression="gzip", compression_opts=9);
    nimg = OpenEXR.InputFile(os.path.join(imgp,'model_normalized_r%d_normal.exr0001.exr'%angle));
    nr,ng,nb = nimg.channels("RGB");
    nr = np.fromstring(nr,dtype=np.float32).reshape((448,448));
    ng = np.fromstring(ng,dtype=np.float32).reshape((448,448));
    nb = np.fromstring(nb,dtype=np.float32).reshape((448,448));
    h5fo.create_dataset("norm448",data=np.stack([nr,ng,nb],axis=2),compression="gzip", compression_opts=9);

def pack(pnpath,partpath,spnobjpath,opath,id,angle):
    partp = os.path.join(partpath,'part_r%d'%angle);
    imgp = spnobjpath;
    data = read_ply(os.path.join(pnpath,'point_sample','ply-10000.ply'));
    label = np.loadtxt(os.path.join(pnpath,'point_sample','label-10000.txt'),dtype=np.int32);
    plypts = np.array(data['points'])[:,:3];
    start = np.min(label);
    end = np.max(label);
    cnt = 0;
    msklst = [];
    smsklst = [];
    ps = [];
    pstouch = [];
    r1 = R.from_euler('x',-90,degrees=True);
    for i in range(start,end+1):
        pv = plypts[label== i,:3].astype(np.float32);
        num = pv.shape[0];
        if num > 0 :
            mskp = 'p_%d_b_msk0001.png'%(cnt,angle);
            msk = Image.open(os.path.join(partpath,'all_msk',mskp));
            msk = np.array(msk).astype(np.float32) / 255.0;
            msk = msk[:,:,2];
            smsk = Image.open(os.path.join(partp,'self_msk',mskp));
            smsk = np.array(smsk).astype(np.float32) / 255.0;
            smsk = smsk[:,:,2];
            if np.sum(msk) > 9:
                pstouch.append(pv);
                cpath = os.path.join(partp,'p_%d_b.ply'%cnt);
                pts = read_ply(cpath);
                pvp = np.array(pts['points'])[:,3].astype(np.float32);
                pvp = r1.apply(pvp);
                ps.append(pvp);
            cnt += 1;
    #print(len(ps));
    num = len(ps);

    obblst = [];
    obbp = [];
    obbf = [];
    obbcnt = 0;
    for pts in ps:
        obba = OBB.build_by_trimesh(pts);
        obbb = OBB.build_from_points(pts);
        if (obba is None) or ((obbb is not None) and (obba.volume > obbb.volume)):
            obbr = obbb;
        else:
            obbr = obba;
        #print('size:',obba.volume);
        obblst.append( obbr );
        obbf.append(bf + obbcnt*8);
        obbcnt += 1;
    #print('obbf:',len(obbf));
    #print('obbcnt:',obbcnt);
    mm = [];
    for pi in range(num-1):
        for pj in range(pi+1,num):
            da,db = ( pstouch[pi],pstouch[pj] ) if obblst[pi].volume > obblst[pj].volume else ( pstouch[pj], pstouch[pi]);
            tree = scipy.spatial.KDTree(da);
            dsta,idxa = tree.query(da,k=2);
            dstb,idxb = tree.query(db,k=1);
            if np.min(dstb) < np.mean(dsta[:,1]):
                mm.append(np.array([pi,pj],dtype=np.int32));

    print('mm:',len(mm));
    if len(mm) < 1:
        return;
    img = Image.open(os.path.join(imgp,'model_normalized_r%d_e.png'%angle));
    img = np.array(img).astype(np.float32) / 255.0;
    h5fo = h5py.File(os.path.join(opath,id+'_r%d.h5'%angle),'w');
    h5fo.create_dataset("img",data=img,compression="gzip", compression_opts=9);
    packorigin(imgp,angle,h5fo);
    h5fo.create_dataset("touch",data=np.stack(mm,axis=0),compression="gzip", compression_opts=9);
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
    print(mm,file=open(os.path.join(partpath,'mm_r%d.txt'%angle),'w'));
    obox = os.path.join(partpath,'box_r%d.ply'%angle);
    write_ply(obox,points=pd.DataFrame(obbv.astype(np.float32)),faces=pd.DataFrame(face),as_text=True);
    h5fo.close();