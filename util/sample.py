import numpy as np;
import torch;
from scipy.spatial import ConvexHull;

def triangulateSphere(pts):
    hull_list = [];
    for i in range(pts.shape[0]):
        pt = pts[i,...];
        hull = ConvexHull(pt.transpose(1,0));
        for j in range(hull.simplices.shape[0]):
            simplex = hull.simplices[j,:];
            triangle = pt[:,simplex];
            m = triangle[:,0];
            p0p1 = triangle[:,1] -  triangle[:,0];
            p1p2 = triangle[:,2] -  triangle[:,1];
            k = np.cross(p0p1,p1p2);
            if np.dot(m,k) < 0:
                tmp = hull.simplices[j,1];
                hull.simplices[j,1] = hull.simplices[j,2];
                hull.simplices[j,2] = tmp;
        hull_list.append(hull);
    return hull_list;

def randsphere2(m=None):
    pts = np.zeros([3,m],np.float32);
    n = np.linspace(1,m,m);
    n += np.random.normal();
    tmp = 0.5*(np.sqrt(5)-1)*n;
    theta = 2.0*np.pi*(tmp - np.floor(tmp));
    pts[0,:] = np.cos(theta);
    pts[1,:] = np.sin(theta);
    pts[2,:] = 2.0*(n - n.min()) / (n.max()-n.min()) - 1.0;
    scale = np.sqrt(1 - np.square(pts[2,:]));
    pts[0,:] *= scale;
    pts[1,:] *= scale;
    return pts;

def area(fv):
    a = np.sqrt(np.sum((fv[:,:,0,:] -  fv[:,:,1,:])**2,axis=2));
    b = np.sqrt(np.sum((fv[:,:,1,:] -  fv[:,:,2,:])**2,axis=2));
    c = np.sqrt(np.sum((fv[:,:,2,:] -  fv[:,:,0,:])**2,axis=2));
    s = (a + b + c) / 2;
    return np.sqrt(s*(s-a)*(s-b)*(s-c));
    
def tri2pts(ver,fidx,n):
    fv = ver[fidx,:].contiguous();
    w = np.abs(np.random.normal(0,1,(fv.shape[0],n,3,1))).astype(np.float32);
    w += 1e-9;
    w = w / np.sum(w,axis=-2,keepdims=True);
    res = fv.view(fv.size(0),1,fv.size(1),fv.size(2))*torch.from_numpy(w);
    res = torch.sum(res,dim=-2);
    res = res.view(-1,3);
    return res;

def tri2pts_batch(ver,fidx,n):
    fv = ver[:,fidx,:].contiguous();
    w = np.abs(np.random.normal(0,1,(1,fv.shape[1],n,3,1))).astype(np.float32);
    w += 1e-9;
    w = w / np.sum(w,axis=-2,keepdims=True);
    w = torch.from_numpy(w).type(fv.type());
    fv = fv.view(fv.size(0),fv.size(1),1,fv.size(2),fv.size(3));
    res = fv*w;
    res = torch.sum(res,dim=-2);
    res = res.view(res.size(0),-1,3);
    return res; 
    
def run(**kwargs):
    import matplotlib.pyplot as plt;
    from mpl_toolkits.mplot3d import Axes3D
    from .data.gen_toybox import box_vert,box_face;
    ver = box_vert[0:3,...];
    ver = torch.from_numpy(ver);
    fidx = box_face;
    v = tri2pts_batch(ver,fidx,100);
    v = v.data.cpu().numpy();
    fig = plt.figure();
    ax = fig.add_subplot(111,projection='3d');
    for  i in range(v.shape[0]):
        ax.scatter(v[i,:,0],v[i,:,1],v[i,:,2],marker='o');
    plt.show();
