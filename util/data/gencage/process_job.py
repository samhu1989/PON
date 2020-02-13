import zipfile as zpf;
import os;
import numpy as np;
import sys;
from scipy.spatial.transform import Rotation as R;
import usesstk;
import useblender;
import OpenEXR, Imath, cv;

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
    #get depth img
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dimg = OpenEXR.InputFile(depth);
    dw = dimg.header()['dataWindow'];
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1);   
    ddata = golden.channel('Z',pt)
    dimg = cv.CreateMat(size[1], size[0], cv.CV_32FC1)
    cv.SetData(dimg,ddata);
    normdimg = np.zeros((size[1], size[0]))
    cv.normalize(dimg,normdimg,0,255,cv.NORM_MINMAX);
    dedge = auto_canny(normdimg.astype(np.uint8));
    #get normal img
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    nimg = OpenEXR.InputFile(norm);
    rCh = nimg.extract_channels();
    ndata = np.asarray(rCh);
    normnimg = np.zeros((size[1], size[0]));
    cv.normalize(np.mean(dimg,axis=2),normnimg,0,255,cv.NORM_MINMAX);
    nedge = auto_canny(normnimg.astype(np.uint8));
    #
    edge = np.bitwise_or(dedge,nedge);
    rgbimg = cv.imread(rgb);
    out = rgbimg.copy();
    out[edge>0,0] = 0.0;
    out[edge>0,1] = 0.0;
    out[edge>0,2] = 0.0;
    cv2.imwrite(rgbo,out);

            
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
            