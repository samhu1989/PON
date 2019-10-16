from __future__ import print_function
import numpy as np;
from functools import cmp_to_key;
from scipy.spatial.transform import Rotation as R;
from scipy.spatial import Delaunay;
from scipy.spatial import ConvexHull;
from ..tools import repeat_face
from .ply import write_ply;
import pandas as pd;
import os;
import h5py;
import json;
from json import JSONEncoder;
from .gen_toybox import write_env,write_env_msk,Encoder;
import matplotlib.pyplot as plt;
from .obb import OBB;
from .gen_toybox import box_face as tri;

c1 = {
4:'chair/chair_seat'
}
t1 = {
1:'table/game_table/ping_pong_table/tabletop',
3:'table/game_table/pool_table/tabletop',
8:'table/regular_table/tabletop'
};
c2 = {
19:'chair/chair_base/regular_leg_base/leg',
13:'chair/chair_base/star_leg_base/central_support'
};
t2 = {
3:'table/game_table/ping_pong_table/table_base/regular_leg_base/leg',
7:'table/game_table/pool_table/table_base/regular_leg_base/leg',
9:'table/picnic_table/regular_table/table_base/regular_leg_base/leg',
16:'table/regular_table/table_base/star_leg_base/central_support',
19:'table/regular_table/table_base/regular_leg_base/leg',
27:'table/regular_table/table_base/drawer_base/leg',
40:'table/regular_table/table_base/pedestal_base/central_support'
};

def zup(env):
    from scipy.spatial.transform import Rotation as R;
    r = R.from_rotvec(np.pi/2 * np.array([1, 0, 0]));
    for i in range(len(env['box'])):
        pts = env['box'][i][:,:].copy();
        env['box'][i][:,:] = (r.apply(pts)).astype(np.float32)
    return;
        
def norm_env(env):
    min = np.finfo(np.float32).eps;
    for i in range(len(env['box'])):
        mf = np.min( env['box'][i][:,2] );
        if min < mf:
            mf = min;
    t = -mf;
    hull_pts = None;
    for i in range(len(env['box'])):
        env['box'][i][:,2] += t;
        env['t'][i][2] += t;
        if hull_pts is None:
            hull_pts = env['box'][i][:,[0,1]];
        else:
            hull_pts = np.concatenate([hull_pts,env['box'][i][:,[0,1]]],axis=0);
    hull = ConvexHull(hull_pts);
    t = - np.mean(hull_pts[hull.vertices,:],axis=0,keepdims=True);
    for i in range(len(env['box'])):
        env['box'][i][:,[0,1]] += t;
        env['t'][i][[0,1]] += t[0,:];
    s = 0.0;
    for i in range(len(env['box'])):
        mx = np.max(np.sqrt(np.sum(np.square(env['box'][i][:,:]),axis=-1)),axis=0);
        if mx > s:
            s = mx;
    for i in range(len(env['box'])):
        env['box'][i][:,:] /= s;
    

def debug_pts(pts,ptsall):
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R;
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    ax.set_aspect('equal');
    ax.scatter(pts[:,0],pts[:,2],pts[:,1],color='r',marker='o');
    if ptsall.shape[0] == 8:
        ax.plot_trisurf(ptsall[:,0],ptsall[:,2],tri,ptsall[:,1],color=(0,0.0,1.0,0.3));
    else:
        ax.scatter(ptsall[:,0],ptsall[:,2],ptsall[:,1],'b',marker='x');
    plt.show();
    return;
    
def debug_env(env,p0,p1):
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R;
    fig = plt.figure();
    ax = fig.add_subplot(121, projection='3d');
    ax.set_aspect('equal');
    ax.scatter(p0[:,0],p0[:,2],p0[:,1],color='r',marker='o');
    ax.scatter(p1[:,0],p1[:,2],p1[:,1],color='g',marker='x');
    for box in env['box']:
        ax.plot_trisurf(box[:,0],box[:,2],tri,box[:,1],color=(0,0.0,1.0,0.3));
    ax = fig.add_subplot(122, projection='3d');
    ax.set_aspect('equal');
    for box in env['box']:
        ax.plot_trisurf(box[:,0],box[:,2],tri,box[:,1],color=(0,0.0,1.0,1.0));
    plt.show();
    return;
    
def pts2box(pts,env,debug=False):
    obb = OBB.build_from_points(pts);
    if debug:
        debug_pts(pts,np.array(obb.points));
    env['idx'].append(-1);
    env['box'].append(np.array(obb.points));
    env['base'].append((obb.max[0]-obb.min[0])*(obb.max[2]-obb.min[2]));
    env['R'].append(obb.rotation);
    env['t'].append(obb.centroid);
    return;

def getpts(h5,bn,iobj,debug=False):
    pts_all = np.array(h5['pts'][iobj,...]);
    msk_gt  = np.array(h5['gt_mask'][iobj,...]);
    msk_gt_lbl  = np.array(h5['gt_mask_label'][iobj,...]);
    msk_gt_vld  = np.array(h5['gt_mask_valid'][iobj,...]);
    lbl = {};
    for idx in range(msk_gt_lbl.shape[0]):
        if msk_gt_vld[idx]:
            cid = msk_gt_lbl[idx];
            if debug:
                print(cid);
            if cid in bn.keys():
                if debug:
                    print(bn[cid])
                if cid in lbl.keys():
                    lbl[cid].append(msk_gt[idx]);
                else:
                    lbl[cid] = [msk_gt[idx]];
    select_msk = None;
    if debug:
        print(lbl.keys());
    for k,v in lbl.items():
        if select_msk is None:
            select_msk = v[np.random.randint(0,len(v))];
        else:
            select_msk = np.logical_or(select_msk,v[np.random.randint(0,len(v))]);
    if debug and select_msk is not None:
        debug_pts(pts_all[select_msk==True,:],pts_all[select_msk==False]);
    if select_msk is None:
        return None;
    else:
        return pts_all[select_msk==True,:];

def gen(p0,p1,debug=False):
    env={'idx':[],'box':[],'top':[0,1],'base':[],'R':[],'t':[]};
    pts2box(p0,env,debug=debug);
    pts2box(p1,env,debug=debug);
    if debug:
        debug_env(env,p0,p1);
    zup(env);
    norm_env(env);
    if debug:
        debug_env(env,p0,p1);
    return env;

def run(**kwargs):
    num = kwargs['nepoch'];
    data_root = kwargs['data_path'];
    debug = False;
    if 'debug' == kwargs['user_key']:
        debug = True;
    if data_root.endswith('train'):
        ref_root = data_root.replace('train','Chair-1');
        p1 = c1;
        p2 = c2;
    if data_root.endswith('test'):
        ref_root = data_root.replace('test','Table-1');
        p1 = t1;
        p2 = t2;
    fs = os.listdir(ref_root);
    refs = [];
    for f in fs:
        if f.endswith('.json'):
            refs.append(os.path.join(ref_root,f));
    #select box with possible repeat in fact
    i = 0;
    while i < num:
        ref = refs[np.random.randint(0,len(refs),dtype= np.int32)];
        rdict = json.load(open(ref,'r'));
        hfn = ref.replace('.json','.h5')
        h51 = h5py.File(hfn,'r');
        if data_root.endswith('train'):
            h52 = h5py.File(hfn.replace('Chair-1','Chair-2'),'r');
        else:
            h52 = h5py.File(hfn.replace('Table-1','Table-2'),'r');
        iobj = np.random.randint(0,len(rdict),dtype= np.int32);
        print(i,':',hfn,'-',iobj);
        robj = rdict[iobj];
        pts0 = getpts(h51,p1,iobj);
        pts1 = getpts(h52,p2,iobj);
        if (pts0 is None) or (pts1 is None):
            continue;
        env = gen(pts0,pts1,debug=debug);
        json.dump(env,open(os.path.join(data_root,'%04d.json'%i),'w'),cls=Encoder);
        write_env(os.path.join(data_root,'%04d.ply'%i),env);
        write_env_msk(os.path.join(data_root,'%04d'%i),env);
        i += 1;
    return;