import zipfile as zpf;
import os;
import numpy as np;
import sys;
from scipy.spatial.transform import Rotation as R;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';
tmproot = '/cephfs/siyu/cage/tmp';

def do_one(job):
    id = job.split('/')[2];
    objpath = os.path.join(tmproot,id,job,'sp/models/model_normalized.obj');
    align_obj_with_4_random_rot(inobj);
    
def align_obj_with_N_random_rot(inobj,N=4):
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
            