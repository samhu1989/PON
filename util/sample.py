import numpy as np;
import torch;

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
