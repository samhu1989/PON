import zipfile as zpf;
import os;
import numpy as np;
import sys;
from scipy.spatial.transform import Rotation as R;
import usesstk;
import useblender;
import OpenEXR, Imath;
import cv2 as cv;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';
tmproot = '/cephfs/siyu/cage/tmp';

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)
	# return the edged image
	return edged

def do_one(job):
    id = job.split('/')[2];
    objpath = os.path.join(tmproot,id,job,'sp/models/model_normalized.obj');
    align_obj_with_N_random_rot(objpath);
    render_obj(objpath);
    render_depth_normal(objpath);
    highlight_edge(objpath);
    
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
    print(ddata.shape);
    print('ddata:',np.min(ddata),np.max(ddata));
    ddata = ddata.reshape((448,448));
    ddata[rgbimg[:,:,3]==0] = 0.0;
    normdimg = np.zeros((448,448))
    normdimg = cv.normalize(ddata,normdimg,0,255,cv.NORM_MINMAX);
    dedge = auto_canny(normdimg.astype(np.uint8));
    #get normal img
    nimg = OpenEXR.InputFile(norm);
    nr,ng,nb = nimg.channels("RGB");
    ndata = 0.2989 * np.fromstring(nr,dtype=np.float32) + 0.5870 * np.fromstring(ng,dtype=np.float32) + 0.1140 * np.fromstring(nb,dtype=np.float32);
    print('ndata:',np.min(ndata),np.max(ndata));
    normnimg = np.zeros((448, 448));
    normnimg = cv.normalize(ndata,normnimg,0,255,cv.NORM_MINMAX);
    nedge = auto_canny(normnimg.astype(np.uint8));
    #
    cv.imwrite(objpath.replace('.obj','_dedge.png'),dedge);
    cv.imwrite(objpath.replace('.obj','_nedge.png'),nedge);
    edge = np.bitwise_or(dedge,nedge);
    cv.imwrite(objpath.replace('.obj','_edge.png'),edge);
    out = rgbimg.copy();
    out[edge>0,0] = 0.0;
    out[edge>0,1] = 0.0;
    out[edge>0,2] = 0.0;
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
    for v in range(N):
        angle = select[v]*360//24; 
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
    return;
            