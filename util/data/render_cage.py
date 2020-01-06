import numpy as np;
import os;
import json;
import h5py;
import shutil;
from .ply import read_ply,write_ply;
import pandas as pd;
import scipy;
from scipy.spatial.transform import Rotation as R;
from scipy.spatial import KDTree
from functools import partial
from PIL import Image;
from .obb import OBB;
from .gen_toybox import box_face as bf;
from util.tools import label_pts;

op = './data/cage/'
pth = os.path.dirname(os.path.abspath(__file__));

def render(path):
    out = os.path.dirname(path);
    cmd = 'blender --background --python %s '%(pth+os.sep+'render_b.py')
    cmd += '-- --output_folder %s --views 1 '%os.path.dirname(os.path.abspath(path))
    cmd += '--obj %s'%os.path.abspath(path);
    os.system(cmd);
    return;
    
def align_pts(path):
    tounit(path,path.replace('.ply','_align.ply'));
    objpath = os.path.join(os.path.dirname(path),'sp','models','model_normalized.obj')
    align_obj(objpath,objpath.replace('.obj','_align.obj'));
    return;
    
def align_obj(inobj,outobj):
    print(inobj);
    pv = [];
    with open(inobj,'r') as fin:
        for line in fin:
            if line.startswith('v '):
                nums = line.split(' ');
                pv.append([float(nums[1]),float(nums[2]),float(nums[3])]);
    pts = np.array(pv);
    mmx = np.max(pts,axis=0);
    mmn = np.min(pts,axis=0);
    pts -= ((mmx + mmn) / 2.0)[np.newaxis,:];
    scale = float(np.max( np.max(pts,axis=0) - np.min(pts,axis=0)));
    pts /= scale;
    i = 0;
    with open(inobj,'r') as fin:
        with open(outobj,'w') as fout:
            for line in fin:
                if line.startswith('v '):
                    nums = line.split(' ');
                    fout.write(nums[0]+' '+str(pts[i,0])+' '+str(pts[i,1])+' '+str(pts[i,2])+'\n');
                    i += 1;
                else:
                    fout.write(line);
    if os.path.exists(outobj+'.mtl'):
        os.remove(outobj+'.mtl');
    return;
    
def tounit(inply,oply):
    data = read_ply(inply);
    pts = np.array(data['points']);
    r = R.from_euler('y', 180, degrees=True);
    pts = r.apply(pts);
    mmx = np.max(pts,axis=0);
    mmn = np.min(pts,axis=0);
    pts -= ((mmx + mmn) / 2.0)[np.newaxis,:];
    scale = float(np.max( np.max(pts,axis=0) - np.min(pts,axis=0)));
    pts /= scale;
    write_ply(oply,points=pd.DataFrame(pts.astype(np.float32)));
    
    
def func(a,vidx):
    idx = np.argwhere(vidx==a);
    return idx;
    
    
def partply(label,obj,ply,opath):
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    start = np.min(label);
    end = np.max(label);
    data = read_ply(ply);
    plypts = np.array(data['points']);
    print(plypts.shape);
    pv = [];
    pf = [];
    with open(obj,'r') as fin:
        for line in fin:
            if line.startswith('v '):
                nums = line.split(' ');
                pv.append([float(nums[1]),float(nums[2]),float(nums[3])]);
            elif line.startswith('f '):
                fidx = line.split(' ');
                f = [];
                for fi in range(2,5):
                    if '/' in fidx[fi]:
                        f.append(int(fidx[fi].split('/')[0])-1);
                    else:
                        f.append(int(fidx[fi])-1);
                pf.append(f);
    objpv = np.array(pv).astype(np.float32);
    objpf = np.array(pf).astype(np.int32);
    tree = KDTree(data=plypts);
    fc = np.mean(objpv[objpf,:],axis=1);
    dst , idx = tree.query(fc,k=1);
    start = np.min(label);
    end = np.max(label);
    print(start,end);
    cnt = 0;
    for i in range(start,end+1):
        partfi = (label[idx] == i);
        num = np.sum(partfi);
        if num > 0 :
            partface = objpf[partfi,:].flatten();
            vidx = np.unique(partface);
            partptsi = objpv[vidx,:];
            res = map(partial(func,vidx=vidx),partface.tolist());
            fidx = np.array(list(res)).reshape(-1,3);
            face = np.zeros(shape=[fidx.shape[0]],dtype=T);
            for i in range(fidx.shape[0]):
                face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
            r = R.from_euler('x',90,degrees=True);
            pc = r.apply(partptsi).astype(np.float32);
            rc = pd.DataFrame( np.repeat(np.array([[255,0,0]],dtype=np.uint8),partptsi.shape[0],axis=0) );
            bc = pd.DataFrame( np.repeat(np.array([[0,0,255]],dtype=np.uint8),partptsi.shape[0],axis=0) );
            pc = pd.DataFrame(pc);
            partptsia = pd.concat([pc,rc],axis=1,ignore_index=True);
            partptsib = pd.concat([pc,bc],axis=1,ignore_index=True);
            write_ply(os.path.join(opath,'p_%d_a.ply'%cnt),points=partptsia,faces=pd.DataFrame(face),color=True);
            write_ply(os.path.join(opath,'p_%d_b.ply'%cnt),points=partptsib,faces=pd.DataFrame(face),color=True);
            cnt += 1;
            
def render_msk(path,angle):
    alst = [];
    blst = [];
    for f in os.listdir(path):
        if '_a.ply' in f:
            alst.append(f);
        if '_b.ply' in f:
            blst.append(f);
    for bf in blst:
        cmd = 'blender --background --python %s '%(pth+os.sep+'render_part.py')
        cmd += '-- --output_folder %s --views 1'%os.path.abspath(path)
        cmd += ' --objp %s'%os.path.abspath(path);
        lst = bf;
        for aa in alst:
            if aa == bf.replace('_b','_a'):
                continue;
            else:
                lst += '~' + aa;
        cmd += ' --objs %s'%lst;
        cmd += ' --angle %d'%angle;
        os.system(cmd);
    return;
    

def run(**kwargs):
    op = kwargs['data_path'];
    for root, dirs, files in os.walk(op, topdown=True):
        for name in files:
            '''
            if name == 'pn.ply':
                path = os.path.join(root, name)
                
                mpath = os.path.join(root,'sp','models','model_normalized_align.obj')
                mpath = os.path.join(root,'sp','models','model_normalized_align.obj')
                dir = os.path.dirname(mpath)
                lst = os.listdir(dir);
                done = False;
                for f in lst:
                    if f.endswith('.png') and f.startswith('model_normalized_align') and ('.obj_' in f):
                        #os.remove(os.path.join(dir,f));
                        done = True;
                if not done:
                    align_pts(path);                
                    render(mpath);
                    render(os.path.join(dir, 'model_normalized.obj'));
                else:
                    continue;
            '''
            '''
            if name == 'pn_align.ply':
                print(root);
                h5f = h5py.File(os.path.join(root,'pn.h5'),'r');
                label = np.array(h5f['label']);
                plypath = os.path.join(root, name);
                objpath = os.path.join(root,'sp','models','model_normalized_align.obj');
                opath = os.path.join(root,'parts');
                if not os.path.exists(opath):
                    os.mkdir(opath);
                else:
                    continue;
                partply(label,objpath,plypath,opath);
                imgp = os.path.join(root,'sp','models');
                imgs = os.listdir(imgp);
                angle = None;
                for fs in imgs:
                    if fs.endswith('.png') and 'model_normalized_align.obj' in fs:
                        angle = int((fs.split('_')[3]).split('.')[0]);
                print(angle)
                render_msk(opath,angle);
            '''
            if name == 'pn_align.ply':
                opath = os.path.dirname(os.path.join(root, name))+'.h5';
                if os.path.exists(opath):
                    continue;
                obox = os.path.join(root, 'box.ply');
                print(opath);
                if os.path.exists(opath):
                    continue;
                h5fi = h5py.File(os.path.join(root,'pn.h5'),'r');
                label = np.array(h5fi['label']);
                #
                plypath = os.path.join(root,name);
                data = read_ply(plypath);
                plypts = np.array(data['points']);
                
                #
                h5f = h5py.File(opath,'w');
                imgp = os.path.join(root,'sp','models');
                imgs = os.listdir(imgp);
                for fs in imgs:
                    if fs.endswith('.png') and 'model_normalized_align.obj' in fs:
                        angle = int((fs.split('_')[3]).split('.')[0]);
                        img = Image.open(os.path.join(imgp,fs));
                        img = np.array(img).astype(np.float32) / 255.0;
                r = R.from_euler('y',-angle,degrees=True);
                plypts = r.apply(plypts);
                plypts = plypts.astype(np.float32);
                label_pts(os.path.join(root,'pn_color.ply'),plypts,label);
                
                #
                h5f.create_dataset("img", data=img[:,:,:3],compression="gzip", compression_opts=9);
                start = np.min(label);
                end = np.max(label);
                print(start,end);
                cnt = 0;
                partp = os.path.join(root,'parts');
                ps = [];
                msklst = [];
                try:
                    for i in range(start,end+1):
                        pv = plypts[label== i,:];
                        num = pv.shape[0];
                        if num > 0 :
                            mskp = 'p_%d_b_%d_msk0001.png'%(cnt,angle);
                            mskpb = 'p_%d_b_%d_mskb0001.png'%(cnt,angle);
                            msk = Image.open(os.path.join(partp,mskp));
                            msk = np.array(msk).astype(np.float32) / 255.0;
                            msk = msk[:,:,2];
                            if np.sum(msk) > 9:
                                ps.append(np.array(pv));
                                m = Image.fromarray(np.array(msk*255.0).astype(np.uint8),mode='L');
                                m.save(os.path.join(partp,mskpb));
                                msklst.append(msk);
                            cnt += 1;
                except:
                    pass;
                #
                if len(msklst) < 2:
                    h5f.close();
                    h5fi.close();
                    os.remove(opath);
                    continue;
                #
                msks = np.stack(msklst,axis=0);
                print('msks',msks.shape);
                h5f.create_dataset("msk", data=msks,compression="gzip", compression_opts=9);
                #
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
                    obblst.append( obbr );
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
                    h5f.close();
                    h5fi.close();
                    os.remove(opath);
                    continue;
                h5f.create_dataset("touch",data=np.stack(mm,axis=0),compression="gzip", compression_opts=9);
                obbk = [];
                for obb in obblst:
                    obbp.append(obb.points);
                    obbk.append(obb.tov);
                obbv = np.concatenate(obbp,axis=0);
                h5f.create_dataset("box",data=np.stack(obbk,axis=0),compression="gzip", compression_opts=9);
                fidx = np.concatenate(obbf,axis=0);
                T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
                face = np.zeros(shape=[12*len(obbf)],dtype=T);
                for i in range(fidx.shape[0]):
                    face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
                write_ply(obox,points=pd.DataFrame(obbv.astype(np.float32)),faces=pd.DataFrame(face));    
                h5f.close();
                h5fi.close();
                #exit();

            
