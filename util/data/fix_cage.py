import zipfile as zpf;
import os;
import h5py;
from .obb import OBB;
import numpy as np;
from ..tools import writebox;
from .ply import read_ply,write_ply;
import io;
import pandas as pd;
from scipy.spatial.transform import Rotation as R;
cagenet = 'cagenet.zip'
partnet = 'partnet.zip' 

def tounit_param(input_pts):
    r = R.from_euler('y', 180, degrees=True);
    pts = r.apply(input_pts);
    mmx = np.max(pts,axis=0);
    mmn = np.min(pts,axis=0);
    center = ((mmx + mmn) / 2.0)[np.newaxis,:];
    pts -= center;
    scale = float(np.max( np.max(pts,axis=0) - np.min(pts,axis=0)));
    pts /= scale;
    return center,scale,pts.astype(np.float32);

def run(**kwargs):
    data_path = kwargs['data_path'];
    with zpf.ZipFile(os.path.join(data_path,cagenet),'r') as cagezip:
        with zpf.ZipFile(os.path.join(data_path,partnet),'r') as partzip:
            for h5name in cagezip.namelist():
                if h5name.endswith('.h5'):
                    print(h5name);
                    cat = os.path.basename(os.path.dirname(h5name));
                    set = os.path.basename(os.path.dirname(os.path.dirname(h5name)));
                    opath = os.path.join(data_path,set,cat);
                    if not os.path.exists(opath):
                        os.makedirs(opath);
                    name = os.path.basename(h5name);
                    h5out = h5py.File(os.path.join(opath,name),'w');
                    id = name.split('_')[0];
                    rd = int(os.path.basename(h5name).split('_r')[1].split('.')[0]);
                    print(id);
                    print(rd);
                    with cagezip.open(h5name) as cagef:
                        cageh5 = h5py.File(cagef,'r');
                        h5out.create_dataset("box3d", data=cageh5['box3d'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("img", data=cageh5['img'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("dmap", data=cageh5['dmap'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("nmap", data=cageh5['nmap'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("msk", data=cageh5['msk'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("smsk", data=cageh5['smsk'],chunks=True,compression="gzip", compression_opts=9);
                        h5out.create_dataset("touch", data=cageh5['touch'],chunks=True,compression="gzip", compression_opts=9);

                        
                    with partzip.open('partnet/'+id+'/point_sample/pts-10000.txt','r') as ptsf:
                        points = np.loadtxt(io.BytesIO(ptsf.read()),dtype=np.float32,delimiter=' ');
                        
                    _,_,points = tounit_param(points);
                    r = R.from_euler('y', rd, degrees=True);
                    points = r.apply(points).astype(np.float32);
                    
                    h5out.create_dataset("pts", data=points,chunks=True,compression="gzip", compression_opts=9);
                        
                    with partzip.open('partnet/'+id+'/point_sample/label-10000.txt','r') as labelf:
                        label = np.loadtxt(io.BytesIO(labelf.read()),dtype=np.int32);
                        
                    h5out.create_dataset("lbl", data=label,chunks=True,compression="gzip", compression_opts=9);