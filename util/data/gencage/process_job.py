import zipfile as zpf;
import os;
import numpy as np;
import sys;
from scipy.spatial.transform import Rotation as R;
import usesstk;
import useblender;
import OpenEXR, Imath;
import cv2 as cv;
from functools import partial
from ply import read_ply, write_ply;
import pandas as pd;
import json;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';
tmproot = '/cephfs/siyu/cage/tmp';

def do_one(job):
    id = job.split('/')[2];
    objpath = os.path.join(tmproot,id,job,'sp/models/model_normalized.obj');
    #
    angles = align_obj_with_N_random_rot(objpath);
    render_obj(objpath);
    render_depth_normal(objpath);
    highlight_edge(objpath);
    #
    partnetpath = os.path.join(tmproot,id,'partnet',id);
    partoutpath = os.path.join(tmproot,id,'part');
    gen_parts(partnetpath,angles,partoutpath);
    
def func(a,vidx):
    idx = np.argwhere(vidx==a);
    return idx;
    
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
    #load only face referenced pv:
    vidx = np.unique(objpf.flatten());
    vout = objpv[vidx,:];
    res = map(partial(func,vidx=vidx),objpf.flatten().tolist());
    fout = np.array(list(res)).reshape(-1,3);
    return vout,fout;
    
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
    
def partply(partpath,rot,opath):
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    data = read_ply(os.path.join(partpath,'point_sample','ply-10000.ply'));
    label = np.loadtxt(os.path.join(partpath,'point_sample','label-10000.txt'),dtype=np.int32);
    plypts = np.array(data['points'])[:,:3];
    start = np.min(label);
    end = np.max(label);
    part_map = json.load(open(os.path.join(partpath,'result_map.json'),'r'));
    partv_lst = [];
    partf_lst = [];
    for i in range(start,end+1):
        num = np.sum(label==i);
        if num > 0 :
            pv = [];
            pf = [];
            pvn = 0;
            for name in part_map['%d'%i]['objs']:
                pvi,pfi = read_obj(os.path.join(partpath,'objs',name+'.obj'));
                pv.append(pvi);
                pf.append(pfi+pvn);
                pvn += pvi.shape[0];
            if len(pv) > 1:
                partv_lst.append(np.concatenate(pv,axis=0));
                partf_lst.append(np.concatenate(pf,axis=0));
            else:
                partv_lst.append(pv[0]);
                partf_lst.append(pf[0]);

    partpts = np.concatenate(partv_lst,axis=0);
    center,scale = tounit_param(partpts);
    for parti in range(len(partv_lst)):
        partptsi = partv_lst[parti];
        partface = partf_lst[parti];
        r = R.from_euler('y',0,degrees=True);
        pc = r.apply(partptsi).astype(np.float32);
        pc -= center;
        pc /= scale;
        r = R.from_euler('y',rot,degrees=True);
        pc = r.apply(pc);
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
    
def gen_parts(partnetpath,angles,pout):
    for angle in angles:
        copath = os.path.join(pout,'part_r%d'%angle);
        if not os.path.exists(copath):
            os.makedirs(copath);
        partply(partnetpath,angle,copath);
        useblender.render_msk(copath);
        
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image);
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged
    
def render_obj(objpath):
    path = os.path.dirname(objpath);
    for f in os.listdir(path):
        if 'model_normalized_r' in f and f.endswith('.obj'):
            usesstk.render(os.path.join(path,f),os.path.abspath('./sstk.json'));
            
def render_depth_normal(objpath):
    path = os.path.dirname(objpath);
    for f in os.listdir(path):
        if 'model_normalized_r' in f and f.endswith('.obj'):
            useblender.render(os.path.join(path,f));
            
def add_edge(objpath):
    rgb = objpath.replace('.obj','.png');
    depth = objpath.replace('.obj','_depth.exr0001.exr');
    norm = objpath.replace('.obj','_normal.exr0001.exr');
    rgbo = objpath.replace('.obj','_e.png');
    rgbimg = cv.imread(rgb,cv.IMREAD_UNCHANGED);
    #get depth img
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dimg = OpenEXR.InputFile(depth);
    dr,dg,db = dimg.channels("RGB");
    ddata = 0.2989 * np.fromstring(dr,dtype=np.float32) + 0.5870 * np.fromstring(dg,dtype=np.float32) + 0.1140 * np.fromstring(db,dtype=np.float32);
    #print(ddata.shape);
    #print('ddata:',np.min(ddata),np.max(ddata));
    ddata = ddata.reshape((448,448));
    ddata[rgbimg[:,:,3]==0] = 0.0;
    normdimg = np.zeros((448,448))
    normdimg = cv.normalize(ddata,normdimg,0,255,cv.NORM_MINMAX);
    dedge = auto_canny(normdimg.astype(np.uint8));
    #get normal img
    nimg = OpenEXR.InputFile(norm);
    nr,ng,nb = nimg.channels("RGB");
    ndata = 0.2989 * np.fromstring(nr,dtype=np.float32) + 0.5870 * np.fromstring(ng,dtype=np.float32) + 0.1140 * np.fromstring(nb,dtype=np.float32);
    #print('ndata:',np.min(ndata),np.max(ndata));
    ndata = ndata.reshape((448,448));
    normnimg = np.zeros((448,448));
    normnimg = cv.normalize(ndata,normnimg,0,255,cv.NORM_MINMAX);
    nedge = auto_canny(normnimg.astype(np.uint8));
    #cv.imwrite(objpath.replace('.obj','_dedge.png'),dedge);
    #cv.imwrite(objpath.replace('.obj','_nedge.png'),nedge);
    edge = np.bitwise_or(dedge,nedge);
    #cv.imwrite(objpath.replace('.obj','_edge.png'),edge);
    out = rgbimg.copy();
    out[edge>0,0] = 0.0;
    out[edge>0,1] = 0.0;
    out[edge>0,2] = 0.0;
    out = cv.resize(out, (224,224), interpolation = cv.INTER_CUBIC)
    cv.imwrite(rgbo,out);
            
def highlight_edge(objpath):
    path = os.path.dirname(objpath);
    for f in os.listdir(path):
        if 'model_normalized_r' in f and f.endswith('.obj'):
            add_edge(os.path.join(path,f));
    
def align_obj_with_N_random_rot(inobj,N=4):
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
    select = np.random.choice(range(24),N, replace=False)
    angles = [];
    for v in range(N):
        angle = select[v]*360//24;
        angles.append(angle);
        r = R.from_rotvec(np.radians(angle) * np.array([0, 1, 0]));
        cpts = r.apply(pts);
        outobj = inobj.replace('.obj','_r%d.obj'%angle);
        i = 0;
        with open(inobj,'r') as fin:
            with open(outobj,'w') as fout:
                for line in fin:
                    if line.startswith('v '):
                        nums = line.split(' ');
                        fout.write(nums[0]+' '+str(cpts[i,0])+' '+str(cpts[i,1])+' '+str(cpts[i,2])+'\n');
                        i += 1;
                    else:
                        fout.write(line);
    return angles;
            