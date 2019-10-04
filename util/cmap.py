import numpy as np;
cdict = {'red': ((0.0, 0.0, 0.0),

(0.1, 0.5, 0.5),

(0.2, 0.0, 0.0),

(0.4, 0.2, 0.2),

(0.6, 0.0, 0.0),

(0.8, 1.0, 1.0),

(1.0, 1.0, 1.0)),

'green':((0.0, 0.0, 0.0),

(0.1, 0.0, 0.0),

(0.2, 0.0, 0.0),

(0.4, 1.0, 1.0),

(0.6, 1.0, 1.0),

(0.8, 1.0, 1.0),

(1.0, 0.0, 0.0)),

'blue': ((0.0, 0.0, 0.0),

(0.1, 0.5, 0.5),

(0.2, 1.0, 1.0),

(0.4, 1.0, 1.0),

(0.6, 0.0, 0.0),

(0.8, 0.0, 0.0),

(1.0, 0.0, 0.0))}

def interpcmap(x,arr):
    ret = None;
    for i in range(1,len(arr)):
        if ( x < arr[i][0] ) and ( x >= arr[i-1][0] ):
            w1 = x-arr[i-1][0];
            w2 = arr[i][0]-x
            ret =  ( w1*arr[i][2] + w2*arr[i-1][1] )/(w1+w2);
        elif x == arr[i][0]:
            ret =  arr[i][1];
    return ret;

def cmap(x):
    r = 255.0*interpcmap(x[0],cdict['red']);
    g = 255.0*interpcmap(x[0],cdict['green']);
    b = 255.0*interpcmap(x[0],cdict['blue']);
    return r,g,b

def map2color(x):
    return np.apply_along_axis(cmap,1,x);

def colorbar(n):
    return map2color(np.linspace(0.0,1.0,n).reshape(-1,1)).astype(np.uint8);

def minmax(x):
    x = x.astype(np.float32);
    x -= np.min(x);
    x /= np.max(x);
    return x*255.0;

def colorcoord(coord):
    xyz = coord.copy()
    return np.apply_along_axis(minmax,0,xyz).reshape((-1,3)).astype(np.uint8);

color = np.array([
[255,165,0],
[84,139,84],
[255,246,143],
[100,149,237],
[132,112,255],
[0,191,255],#
[0,255,255],
[127,255,212],
[60,179,113],
[255,228,196],
[188,143,143],
[244,164,96],#
[255,192,203],
[230,230,250],
[193,205,193],
[105,139,34],#
[238,238,209],
[255,48,48],
[238,154,0],
[224,102,255],#
[155,48,255],
[144,238,144],
[139,0,0],
[0,139,139],
[176,48,96]
],dtype=np.uint8);
 
def colorindex(coord,num):
    N = coord.shape[0];
    step = N // num;
    idx = np.array([(x // step) for x in range(N)],dtype=np.int32);
    return color[idx,:];
    
def colorbyindex(idx):
    return color[idx,:].reshape(-1,3);
    