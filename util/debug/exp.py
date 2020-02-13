import numpy as np;

def run(**kwargs):
    x = np.abs(np.random.normal(0.0,1.0,[1000,3]));
    mx = np.mean(x,axis=0);
    xm = np.max(x,axis=1,keepdims=True);
    xx = x / xm;
    mxx = np.mean(xx,axis=0);
    print(mx);
    print(mxx*np.mean(xm));
    z = 1 / (1 + np.exp(-np.random.normal(0,200,[1000,3])));
    mz = np.mean(z,axis=0);
    zm = np.max(z,axis=1,keepdims=True);
    zz = z / zm;
    mzz = np.mean(zz,axis=0);
    print(mz);
    print(mzz*np.mean(zm));
    