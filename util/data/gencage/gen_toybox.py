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
import json;
from json import JSONEncoder;
box_face = np.array(
[
[1,3,0],
[2,3,1],
[7,5,4],
[7,6,5],
[4,5,0],
[5,1,0],
[1,5,2],
[2,5,6],
[6,3,2],
[6,7,3],
[4,0,3],
[3,7,4]
],
dtype=np.int32
);
box_vert = np.array(
[
#box1
    [
        [-0.3,0.4,0.0],[0.3,0.4,0.0],[0.3,-0.4,0.0],[-0.3,-0.4,0.0],
        [-0.3,0.4,0.2],[0.3,0.4,0.2],[0.3,-0.4,0.2],[-0.3,-0.4,0.2]
    ],
#box2
    [
        [-0.4,0.4,0.0],[0.4,0.4,0.0],[0.4,-0.4,0.0],[-0.4,-0.4,0.0],
        [-0.4,0.4,0.2],[0.4,0.4,0.2],[0.4,-0.4,0.2],[-0.4,-0.4,0.2]
    ],
#box3
    [
        [-0.2,0.2,0.0],[0.2,0.2,0.0],[0.2,-0.2,0.0],[-0.2,-0.2,0.0],
        [-0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,-0.2,0.2],[-0.2,-0.2,0.2]
    ],
#box4
    [
        [-0.1,0.2,0.0],[0.1,0.2,0.0],[0.1,-0.2,0.0],[-0.1,-0.2,0.0],
        [-0.1,0.2,0.2],[0.1,0.2,0.2],[0.1,-0.2,0.2],[-0.1,-0.2,0.2]
    ],
#box5
    [
        [-0.25,0.25,0.0],[0.25,0.25,0.0],[0.25,-0.25,0.0],[-0.25,-0.25,0.0],
        [-0.25,0.25,0.5],[0.25,0.25,0.5],[0.25,-0.25,0.5],[-0.25,-0.25,0.5]
    ]
]
,
dtype=np.float32
);
#box_vert = box_vert*1.0;

box_vol = [0.096,0.128,0.032,0.016,0.125];
box_base = [0.48,0.64,0.16,0.08,0.25];
w = np.linspace(0.0,1.0, num=250);
box_border_w = np.zeros([1000,4,1]);
box_border_w[0:250,0,0] = w;
box_border_w[0:250,1,0] = 1.0 - w;
box_border_w[250:500,1,0] = w;
box_border_w[250:500,2,0] = 1.0 - w;
box_border_w[500:750,2,0] = w;
box_border_w[500:750,3,0] = 1.0 - w;
box_border_w[750:1000,3,0] = w;
box_border_w[750:1000,0,0] = 1.0 - w;


def box_sort_base(x,y):
    return  box_base[y] - box_base[x];
    
def center_env(env):
    hull_pts = np.zeros([1,2]);
    for i in range(len(env['box'])):
        base_pts = env['box'][i][0:4,:];
        if (base_pts[:,2] == 0.0).all():#box on the ground is counted
            hull_pts = np.concatenate([hull_pts,base_pts[:,0:2]],axis=0);
    hull = ConvexHull(hull_pts);
    t = - np.mean(hull_pts[hull.vertices,:],axis=0,keepdims=True);
    for i in range(len(env['box'])):
        env['box'][i][:,0:2] += t;
        env['t'][i][:,0:2] += t;
    
def place(env,idx):
    #if no box in env add the current box add origin with a random rotation along axis z;
    if 0 == len(env['box']):
        env['idx'].append(idx);
        theta = np.random.uniform(0,0.25*np.pi);
        r = R.from_quat([0, 0, np.sin(theta), np.cos(theta)]);
        env['box'].append(r.apply(box_vert[idx,...]));
        env['top'].append(1);
        env['base'].append(box_base[idx]);
        env['R'].append(np.array([0, 0, np.sin(theta), np.cos(theta)]));
        env['t'].append(np.zeros([1,3],np.float32));
    else: 
        # if there is a box larger than current one that is not occupied then place on top
        #count available top:
        cnt = 0;
        for i in range(len(env['box'])):
            if (env['top'][i] == 1) and (env['base'][i] > box_base[idx]):
                cnt += 1;
        if cnt > 0:
            top = int(np.random.uniform(0,cnt));
            for i in range(len(env['box'])):
                if (env['top'][i] == 1) and (env['base'][i] > box_base[idx]):
                    top -= 1;
                    if top > 0:
                        continue;
                    env['top'][i] = 0; #occupy this top;
                    theta = np.random.uniform(0,0.25*np.pi);
                    r = R.from_quat([0, 0, np.sin(theta), np.cos(theta)]);
                    top_vert = env['box'][i][4:8,:];
                    w = np.random.uniform(0.5,1.0,[4,1]);
                    w = w / np.sum(w,keepdims=True);
                    t = np.sum(w*top_vert,axis=0,keepdims=True);
                    #append current box
                    env['idx'].append(idx);
                    env['box'].append(r.apply(box_vert[idx,...])+t);
                    env['top'].append(1);
                    env['base'].append(box_base[idx]);
                    env['R'].append(np.array([0, 0, np.sin(theta), np.cos(theta)]));
                    env['t'].append(t);
        else:#else place on ground
            #randomly rotate current box;
            theta = np.random.uniform(0,0.25*np.pi);
            r = R.from_quat([0, 0, np.sin(theta), np.cos(theta)]);
            cbox = r.apply(box_vert[idx,...]);
            hull_pts = np.zeros([1,2]);
            for i in range(len(env['box'])):
                base_pts = env['box'][i][0:4,:];
                if (base_pts[:,2] == 0.0).all():
                    hull_pts = np.concatenate([hull_pts,base_pts[:,0:2]],axis=0);
            hull = Delaunay(hull_pts);
            pts = np.sum(box_border_w*cbox[0:4,0:2].reshape(1,4,2),axis=1);
            tell = (hull.find_simplex(pts) >= 0).any();
            dt = np.zeros([1,3]);
            dxy = np.random.uniform(0,1.0,[1,2]);
            dxy = 0.01*dxy/np.linalg.norm(dxy,2);
            dt[0,0:2] = dxy;
            t = np.zeros([1,3]);
            while tell:
                t += dt;
                cbox = cbox + dt;
                pts = np.sum(box_border_w*cbox[0:4,0:2].reshape(1,4,2),axis=1);
                tell = (hull.find_simplex(pts) >= 0).any();
            env['idx'].append(idx);
            env['box'].append(cbox);
            env['top'].append(1);
            env['base'].append(box_base[idx]);
            env['R'].append(np.array([0, 0, np.sin(theta), np.cos(theta)]));
            env['t'].append(t);
            center_env(env);
                
def write_env(path,env):
    box_num = len(env['box']);
    fidx = repeat_face(box_face,box_num,8);
    pts = np.zeros([box_num*8,3],dtype=np.float32);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    for i in range(box_num):
        cbox = env['box'][i];
        pts[8*i:8*(i+1),:] = cbox;
    write_ply(path,pd.DataFrame(pts),faces=pd.DataFrame(face),as_text=True);
    
def write_env_msk(path,env):
    box_num = len(env['box']);
    fidx = repeat_face(box_face,box_num,8);
    pts = np.zeros([box_num*8,3],dtype=np.float32);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    for i in range(box_num):
        cbox = env['box'][i];
        pts[8*i:8*(i+1),:] = cbox;
    color = np.zeros(pts.shape,dtype=np.uint8);
    color[:,0] = 255;
    for i in range(box_num):
        c = color.copy();
        c[8*i:8*(i+1),0] = 0;
        c[8*i:8*(i+1),2] = 255;
        ptsc = pd.concat([pd.DataFrame(pts),pd.DataFrame(c)],axis=1,ignore_index=True);
        write_ply(path+'_msk%02d.ply'%i,ptsc,faces=pd.DataFrame(face),as_text=True,color=True);

class Encoder(JSONEncoder):
    def default(self,o):
        if isinstance(o,np.ndarray):
            return o.tolist();
        else:
            return float(o);

def gen(bidx):
    #decide the number of boxes [2,3] 
    bidx = sorted(bidx,key=cmp_to_key(box_sort_base));
    env={'idx':[],'box':[],'top':[],'base':[],'R':[],'t':[]};
    for bi in bidx:
        place(env,bi);
    return env;

def run(**kwargs):
    num = kwargs['nepoch'];
    data_root = kwargs['data_path'];
    bnum = kwargs['batch_size'];
    #select box with possible repeat in fact
    for i in range(num):
        if data_root.endswith('train'):
            bidx = (np.random.uniform(0,4,[bnum]).astype(np.int32)).tolist();
        else:
            bidx = (np.random.uniform(0,5,[bnum]).astype(np.int32)).tolist();
        env = gen(bidx);
        json.dump(env,open(os.path.join(data_root,'%04d.json'%i),'w'),cls=data_handler);
        write_env(os.path.join(data_root,'%04d.ply'%i),env);
        write_env_msk(os.path.join(data_root,'%04d'%i),env);
    return;