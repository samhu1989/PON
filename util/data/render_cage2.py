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
    
def tounit_param(input_pts):
    r = R.from_euler('y', 180, degrees=True);
    pts = r.apply(input_pts);
    mmx = np.max(pts,axis=0);
    mmn = np.min(pts,axis=0);
    center = ((mmx + mmn) / 2.0)[np.newaxis,:];
    pts -= center;
    scale = float(np.max( np.max(pts,axis=0) - np.min(pts,axis=0)));
    pts /= scale;
    return center,scale;
    
def func(a,vidx):
    idx = np.argwhere(vidx==a);
    return idx;
            
def render_msk(path,angle):
    alst = [];
    blst = [];
    for f in os.listdir(path):
        if '_a.ply' in f:
            alst.append(f);
        if '_b.ply' in f:
            blst.append(f);
    selfpath = os.path.join(path,'self_msk');
    if not os.path.exists(selfpath):
        os.mkdir(selfpath);
    allpath = os.path.join(path,'all_msk');
    if not os.path.exists(allpath):
        os.mkdir(allpath);
    for bf in blst:
        cmd = 'blender --background --python %s '%(pth+os.sep+'render_part.py')
        cmd += '-- --output_folder %s --views 1'%os.path.abspath(allpath)
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
    for bf in blst:
        cmd = 'blender --background --python %s '%(pth+os.sep+'render_part.py')
        cmd += '-- --output_folder %s --views 1'%os.path.abspath(selfpath)
        cmd += ' --objp %s'%os.path.abspath(path);
        lst = bf;
        cmd += ' --objs %s'%lst;
        cmd += ' --angle %d'%angle;
        os.system(cmd);
    return;
    
def read_obj(obj):
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
                for fi in range(1,len(fidx)):
                    if '/' in fidx[fi]:
                        f.append(int(fidx[fi].split('/')[0])-1);
                    else:
                        f.append(int(fidx[fi])-1);
                pf.append(f);
    objpv = np.array(pv).astype(np.float32);
    objpf = np.array(pf).astype(np.int32);
    return objpv,objpf;
    
def partply(label,objpath,plypath,opath):
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    data = read_ply(plypath);
    plypts = np.array(data['points']);
    start = np.min(label);
    end = np.max(label);
    print(start,end);
    part_map = json.load(open(os.path.join(objpath,'result_map.json'),'r'));
    partv_lst = [];
    partf_lst = [];
    for i in range(start,end+1):
        num = np.sum(label==i);
        if num > 0 :
            name = part_map['%d'%i]['objs'][0];
            partptsi,partface = read_obj(os.path.join(objpath,'objs',name+'.obj'));
            partv_lst.append(np.array(partptsi));
            partf_lst.append(np.array(partface));
    partpts = np.concatenate(partv_lst,axis=0);
    center,scale = tounit_param(partpts);
    for parti in range(len(partv_lst)):
        partptsi = partv_lst[parti];
        partface = partf_lst[parti];
        r = R.from_euler('y',180,degrees=True);
        pc = r.apply(partptsi).astype(np.float32);
        pc -= center;
        pc /= scale;
        face = np.zeros(shape=[len(partface)],dtype=T);
        for i in range(len(partface)):
            face[i] = (3,int(partface[i][0]),int(partface[i][1]),int(partface[i][2]));
        r = R.from_euler('x',90,degrees=True);
        pc = r.apply(pc).astype(np.float32);
        rc = pd.DataFrame( np.repeat(np.array([[255,0,0]],dtype=np.uint8),partptsi.shape[0],axis=0) );
        bc = pd.DataFrame( np.repeat(np.array([[0,0,255]],dtype=np.uint8),partptsi.shape[0],axis=0) );
        pc = pd.DataFrame(pc);
        partptsia = pd.concat([pc,rc],axis=1,ignore_index=True);
        partptsib = pd.concat([pc,bc],axis=1,ignore_index=True);
        write_ply(os.path.join(opath,'p_%d_a.ply'%parti),points=partptsia,faces=pd.DataFrame(face),color=True);
        write_ply(os.path.join(opath,'p_%d_b.ply'%parti),points=partptsib,faces=pd.DataFrame(face),color=True);
    return;
    
def partplyf(label,objpath,plypath,opath):
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    data = read_ply(plypath);
    plypts = np.array(data['points']);
    start = np.min(label);
    end = np.max(label);
    print(start,end);
    part_map = json.load(open(os.path.join(objpath,'result_map.json'),'r'));
    partv_lst = [];
    partf_lst = [];
    fix = False;
    for i in range(start,end+1):
        num = np.sum(label==i);
        if num > 0 :
            pv = [];
            pf = [];
            pvn = 0;
            for name in part_map['%d'%i]['objs']:
                pvi,pfi = read_obj(os.path.join(objpath,'objs',name+'.obj'));
                pv.append(pvi);
                pf.append(pfi+pvn);
                pvn += pvi.shape[0];
            if len(pv) > 1:
                partv_lst.append(np.concatenate(pv,axis=0));
                partf_lst.append(np.concatenate(pf,axis=0));
                fix = True;
            else:
                partv_lst.append(pv[0]);
                partf_lst.append(pf[0]);
    if not fix:
        return False;
    partpts = np.concatenate(partv_lst,axis=0);
    center,scale = tounit_param(partpts);
    for parti in range(len(partv_lst)):
        partptsi = partv_lst[parti];
        partface = partf_lst[parti];
        r = R.from_euler('y',180,degrees=True);
        pc = r.apply(partptsi).astype(np.float32);
        pc -= center;
        pc /= scale;
        face = np.zeros(shape=[len(partface)],dtype=T);
        for i in range(len(partface)):
            face[i] = (3,int(partface[i][0]),int(partface[i][1]),int(partface[i][2]));
        r = R.from_euler('x',90,degrees=True);
        pc = r.apply(pc).astype(np.float32);
        rc = pd.DataFrame( np.repeat(np.array([[255,0,0]],dtype=np.uint8),partptsi.shape[0],axis=0) );
        bc = pd.DataFrame( np.repeat(np.array([[0,0,255]],dtype=np.uint8),partptsi.shape[0],axis=0) );
        pc = pd.DataFrame(pc);
        partptsia = pd.concat([pc,rc],axis=1,ignore_index=True);
        partptsib = pd.concat([pc,bc],axis=1,ignore_index=True);
        write_ply(os.path.join(opath,'p_%d_a.ply'%parti),points=partptsia,faces=pd.DataFrame(face),color=True);
        write_ply(os.path.join(opath,'p_%d_b.ply'%parti),points=partptsib,faces=pd.DataFrame(face),color=True);
    return True;
    

def run(**kwargs):
    op = kwargs['data_path'];
    uk = kwargs['user_key'];
    for root, dirs, files in os.walk(op, topdown=True):
        for name in files:
            if name == 'pn.ply':
                id = os.path.basename(root);
                if not id.startswith(uk):
                    continue;
                print(root);
                h5f = h5py.File(os.path.join(root,'pn.h5'),'r');
                label = np.array(h5f['label']);
                plypath = os.path.join(root, name);
                objpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(root))),'part',id);
                opath = os.path.join(root,'parts2');
                #if not os.path.exists(opath):
                    #os.mkdir(opath);
                #else:
                    #continue;
                if partplyf(label,objpath,plypath,opath):
                    imgp = os.path.join(root,'sp','models');
                    imgs = os.listdir(imgp);
                    angle = None;
                    for fs in imgs:
                        if fs.endswith('.png') and 'model_normalized_align.obj' in fs:
                            angle = int((fs.split('_')[3]).split('.')[0]);
                    print(angle)
                    render_msk(opath,angle);
                
                

            
