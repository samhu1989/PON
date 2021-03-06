import os
import random
import numpy as np
import torch;
from .data.ply import write_ply;
import scipy;
from scipy.sparse import dok_matrix;
from scipy.sparse import csr_matrix;
from scipy.spatial import ConvexHull;
from scipy.spatial import Delaunay
import pandas as pd;
from .cmap import color;
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

def partial_restore(net,path,keymap={}):
    if torch.cuda.is_available():
        olddict = torch.load(path);
    else:
        olddict = torch.load(path,map_location=torch.device('cpu'));
    #print(olddict.keys());
    mdict = net.state_dict();
    #print(olddict.keys());
    #print(mdict.keys());
    newdict = {};
    for k,v in mdict.items():
        if ( k in olddict ) and ( v.size() == olddict[k].size() ):
            newdict[k] = olddict[k];
        elif k in keymap and keymap[k] in olddict:
            newdict[k] = olddict[keymap[k]];
        else:
            print(k,'in model is not assigned');
    mdict.update(newdict);
    net.load_state_dict(mdict);


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1.0):
        self.val = float(val);
        self.sum += val * n;
        self.count += n;
        self.avg = self.sum / self.count;
        
        
class AvgMeterGroup(object):
    def __init__(self,name):
        self.name = name;
        self.overall_meter = AverageValueMeter();
        self.category_meters = {};
        
    def reset(self):
        self.overall_meter.reset();
        for v in self.category_meters.values():
            v.reset();
            
    def update(self,val,cat):
        if isinstance(val,torch.Tensor):
            val = val.data.cpu().numpy();
        for i,c in enumerate(cat):
            self.overall_meter.update(val[i]);
            if c in self.category_meters.keys():
                self.category_meters[c].update(val[i]);
            else:
                self.category_meters[c] = AverageValueMeter();
                self.category_meters[c].update(val[i]);
                
    def __str__(self):
        ret = 'mean:%10.6f'%self.overall_meter.avg;
        for k,v in self.category_meters.items():
            ret += ','+k+':%10.6f'%v.avg;
        return ret;
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'];
       
writers = {}; 
def write_tfb(tfb_dir,meters,iter,nb,optim):
    from torch.utils.tensorboard import SummaryWriter;
    global writers;
    if tfb_dir in writers.keys():
        writer = writers[tfb_dir];
    else:
        writer = SummaryWriter(log_dir=tfb_dir);
        writers[tfb_dir] = writer;
    for km,meter in meters.items():
        writer.add_scalar(km+'/'+'mean',meter.overall_meter.avg,iter);
        for k,v in meter.category_meters.items():
            writer.add_scalar(km+'/'+k,v.avg,iter);
    writer.add_scalar('lr',get_lr(optim),iter);
    
def write_tfb_loss(tfb_dir,loss,iter,nb,optim):
    from torch.utils.tensorboard import SummaryWriter;
    global writers;
    if tfb_dir in writers.keys():
        writer = writers[tfb_dir];
    else:
        writer = SummaryWriter(log_dir=tfb_dir);
        writers[tfb_dir] = writer;
    for k,v in loss.items():
        writer.add_scalar(k,v.data.cpu().numpy(),iter);
    writer.add_scalar('lr',get_lr(optim),iter);
        
def triangulate(pts):
    hull_list = [];
    for i in range(pts.shape[0]):
        pt = pts[i,...];
        if pt.shape[-1] == 2:
            hull = Delaunay(pt);
        if pt.shape[-1] == 3:
            hull = ConvexHull(pt);
        for j in range(hull.simplices.shape[0]):
            simplex = hull.simplices[j,:];
            triangle = pt[simplex,:];
            m = np.array([0,0,0],dtype=np.float32);
            p0p1 = np.array([0,0,0],dtype=np.float32);
            p1p2 = np.array([0,0,0],dtype=np.float32);
            if pt.shape[-1] == 2:
                m[0:2] = triangle[0,:];
                p0p1[0:2] = triangle[1,:] -  triangle[0,:];
                p1p2[0:2] = triangle[2,:] -  triangle[1,:];
            if pt.shape[-1] == 3:
                m = triangle[0,:];
                p0p1 = triangle[1,:] -  triangle[0,:];
                p1p2 = triangle[2,:] -  triangle[1,:];
            k = np.cross(p0p1,p1p2);
            if np.dot(m,k) < 0:
                tmp = hull.simplices[j,1];
                hull.simplices[j,1] = hull.simplices[j,2];
                hull.simplices[j,2] = tmp;
        hull_list.append(hull);
    return hull_list;
    
def repeat_face(simp,n,num):
    newsimp = np.zeros([simp.shape[0]*n,3],dtype=simp.dtype)
    for i in range(n):
        newsimp[i*simp.shape[0]:(i+1)*simp.shape[0],:] = simp + i*num;
    return newsimp;
    
def genface(pts,num):
    pts_num = pts.shape[0];
    patch_pts_num = pts_num // num;
    p = pts[:patch_pts_num,:];
    fidx = triangulate(p.reshape(1,p.shape[0],p.shape[-1]));
    simp = repeat_face(fidx[0].simplices,num,patch_pts_num);
    return simp.copy();

pts = np.array([
[0,0.8506508,0.5257311], 
[0,0.8506508,-0.5257311], 
[0,-0.8506508,0.5257311], 
[0,-0.8506508,-0.5257311], 
[0.8506508,0.5257311,0  ], 
[0.8506508,-0.5257311,0 ], 
[-0.8506508,0.5257311,0 ],
[-0.8506508,-0.5257311,0],
[0.5257311,0,0.8506508  ], 
[-0.5257311,0,0.8506508 ], 
[0.5257311,0,-0.8506508 ],
[-0.5257311,0,-0.8506508]],dtype=np.float32);

f = np.array([
[1,0,4], 
[0,1,6], 
[2,3,5], 
[3,2,7], 
[4,5,10], 
[5,4,8],
[6,7,9], 
[7,6,11], 
[8,9,2 ],
[9,8,0 ],
[10,11,1 ],
[11,10,3], 
[0,8,4 ],
[0,6,9 ],
[1,4,10 ],
[1,11,6 ],
[2,5,8 ],
[2,9,7 ],
[3,10,5 ],
[3,7,11] 
],dtype=np.int32
)
    
def write_pts2sphere(path,points):
    n = points.shape[0];
    m = pts.shape[0];
    fidx = repeat_face(f,n,m);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    y = 0.01*pts.reshape((1,pts.shape[0],pts.shape[-1])) + points.reshape((points.shape[0],1,points.shape[-1]));
    write_ply(path,points = pd.DataFrame(y.reshape((-1,points.shape[-1]))),faces=pd.DataFrame(face));
    return;
    
def randsphere2(m=100):
    pts = np.zeros([m,3],np.float32);
    n = np.linspace(1,m,m);
    n += np.random.normal();
    tmp = 0.5*(np.sqrt(5)-1)*n;
    theta = 2.0*np.pi*(tmp - np.floor(tmp));
    pts[:,0] = np.cos(theta);
    pts[:,1] = np.sin(theta);
    pts[:,2] = 2.0*(n - n.min()) / (n.max()-n.min()) - 1.0;
    scale = np.sqrt(1 - np.square(pts[2,:]));
    pts[:,0] *= scale;
    pts[:,1] *= scale;
    return pts; 
    
def write_pts2sphere_m(path,points,m=64):
    n = points.shape[0];
    pts = randsphere2(m);
    f = triangulate(pts);
    fidx = repeat_face(f,n,m);
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    y = 0.005*pts.reshape((1,pts.shape[0],pts.shape[-1])) + points.reshape((points.shape[0],1,points.shape[-1]));
    write_ply(path,points = pd.DataFrame(y.reshape((-1,points.shape[-1]))),faces=pd.DataFrame(face));
    return;
    
def merge_mesh(path,dlst,with_face=False):
    points = None;
    colors = None;
    fidx = None;
    num = 0;
    for idxf,d in enumerate(dlst):
        if idxf == 0:
            points = d['points'].to_numpy();
            colors = np.tile(color[idxf,:],points.shape[0]).reshape(points.shape[0],3);
            if with_face:
                fidx = d['mesh'].to_numpy();
        else:
            pts = d['points'].to_numpy();
            points = np.concatenate([points,pts]);
            c = np.tile(color[idxf,:],pts.shape[0]).reshape(pts.shape[0],3)
            colors = np.concatenate([colors,c]);
            if with_face:
                pt_num = dlst[idxf-1]['points'].to_numpy().shape[0];
                num += pt_num;
                f = d['mesh'].to_numpy();
                fidx = np.concatenate([fidx,f+num]);
    pointsc = pd.concat([pd.DataFrame(points),pd.DataFrame(colors)],axis=1,ignore_index=True);
    if with_face:
        T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
        face = np.zeros(shape=[fidx.shape[0]],dtype=T);
        for i in range(fidx.shape[0]):
            face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
        write_ply(path,points = pointsc,faces=pd.DataFrame(face),color=True);
    else:
        write_ply(path,points = pointsc,color=True);
        
        
def label_pts(path,pts,label):
    colors = color[label - np.min(label),:];
    pointsc = pd.concat([pd.DataFrame(pts),pd.DataFrame(colors)],axis=1,ignore_index=True);
    write_ply(path,points = pointsc,color=True);
    
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

def writebox(path,box,colors=None):
    fidx = box_face;
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    bn = len(box);
    face = np.zeros(shape=[bn*fidx.shape[0]],dtype=T);
    for i in range(bn*fidx.shape[0]):
        nn = i // fidx.shape[0];
        ni = i % fidx.shape[0];
        face[i] = (3,fidx[ni,0]+nn*8,fidx[ni,1]+nn*8,fidx[ni,2]+nn*8);
    pts = np.concatenate(box,axis=0);
    if colors is None:
        write_ply(path,points=pd.DataFrame(pts.astype(np.float32)),faces=pd.DataFrame(face));
    else:
        colors = np.concatenate(colors,axis=0);
        pointsc = pd.concat([pd.DataFrame(pts.astype(np.float32)),pd.DataFrame(colors)],axis=1,ignore_index=True);
        write_ply(path,points=pointsc,faces=pd.DataFrame(face),color=True);
        
def func(a):
    idx = np.where(a.flatten() > 0.5 );
    if idx[0].size < 1:
        pix = np.zeros(3).astype(np.uint8);
    else:
        pix = color[idx[0][0],:];
    return pix.flatten();
        
def msk2img(msk):
    return np.apply_along_axis(func, 0, msk);
    
def draw_bounding_box_on_image(image,box,color='red',thickness=1,display_str_list=(),use_normalized_coordinates=False):
    ymin = box[0];
    xmin = box[1];
    ymax = box[2];
    xmax = box[3];
    
    draw = ImageDraw.Draw(image);
    im_width, im_height = image.size;
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24);
    except IOError:
        font = ImageFont.load_default();

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * np.sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle( [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],fill=color)
        draw.text( (left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
        text_bottom -= text_height - 2 * margin
    