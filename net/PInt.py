import torch;
import numpy as np;

def path_integral(map,path):
    return 
    
def path_generate(start,end):
    pts = np.concatenate([start,end],axis=0);
    pts_m = np.mean(pts,keepdims=True,axis=0);
    pts = pts - pts_m;
    u, s, vh = np.linalg.svd(pts, full_matrices=True);
    x = np.matmul(u,pts);
    print(u);
    print(vh);
    print(x);
    return pts_m;                                                                        

def run(**kwargs):
    import matplotlib.pyplot as plt;
    start = np.random.randint(0,224,size=[1,2]);
    end = np.random.randint(0,224,size=[1,2]);
    img = np.ones((224,224)).astype(np.float32);
    plt.imshow(img*255.0,cmap='gray');
    p = path_generate(start,end);
    plt.plot(p[:,0],p[:,1],'*');
    plt.plot(start[:,0],start[:,1],'x');
    plt.plot(end[:,0],end[:,1],'o');
    plt.show();
    return;