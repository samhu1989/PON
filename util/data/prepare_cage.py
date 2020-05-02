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
                    id = os.path.basename(h5name).split('_')[0];
                    rd = int(os.path.basename(h5name).rstrip('.h5').split('_r')[1]);
                    with cagezip.open(h5name) as cagef:
                        cageh5 = h5py.File(cagef,'r');
                        cagebox = np.array(cageh5['box']);
                        box = [];
                        for i in range(cagebox.shape[0]):
                            box.append(OBB.v2points(cagebox[i,...]));
                        writebox('./log/debug_box.ply',box); 
                    with partzip.open('partnet/'+id+'/point_sample/pts-10000.txt','r') as ptsf:
                        points = np.loadtxt(io.BytesIO(ptsf.read()),dtype=np.float32,delimiter=' ');
                        
                    _,_,points = tounit_param(points);
                    r = R.from_euler('y', rd, degrees=True);
                    points = r.apply(points).astype(np.float32);
                        
                    with partzip.open('partnet/'+id+'/point_sample/label-10000.txt','r') as labelf:
                        label = np.loadtxt(io.BytesIO(labelf.read()),dtype=np.int32);
                    
                    write_ply('./log/debug.ply',points=pd.DataFrame(points));
                    print(points.shape,points.dtype);
                    print(label.shape,label.dtype);
                    exit();
    

