import torch;
import numpy as np;

def path_integral(map,path):
    return 
    
def path_generate(start,end):
    pts = np.concatenate([start,end],axis=0);
    pts_m = np.mean(pts,keepdims=True,axis=0);
    print('i',pts);
    pts = pts - pts_m;
    print('c',pts);
    u, s, vh = np.linalg.svd(pts, full_matrices=True);
    xend = np.matmul(vh,pts.T).T;
    print('--',xend)
    num = int(np.abs(xend[1,0]-xend[0,0]));
    x = np.zeros([num,2],dtype=np.float32);
    for i,xtmp in enumerate(np.linspace(xend[0,0],xend[1,0],num)):
        x[i,0] = xtmp;
        x[i,1] = 0.01*(xtmp - xend[0,0])*(xtmp - xend[1,0]) + (xend[0,1]+xend[1,1])/2.0;
    print('xlin',x[0,:],x[-1,:]);
    p = np.matmul(vh.T,x.T).T;
    print('b',p[0,:],p[-1,:]);
    p += pts_m;
    print('r',p[0,:],p[-1,:]);
    return p , x ;                                                                        

def run(**kwargs):
    import matplotlib.pyplot as plt;
    start = np.random.randint(0,224,size=[1,2]);
    end = np.random.randint(0,224,size=[1,2]);
    img = np.ones((224,224)).astype(np.float32);
    p,x = path_generate(start,end);
    plt.subplot(121);
    plt.plot(x[:,0],x[:,1],'o');
    plt.subplot(122);
    plt.imshow(img*255.0,cmap='gray');
    plt.plot(p[:,0],p[:,1],'*');
    plt.plot(start[:,0],start[:,1],'x');
    plt.plot(end[:,0],end[:,1],'o');
    plt.show();
    return;