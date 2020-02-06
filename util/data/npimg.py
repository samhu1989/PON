import numpy as np;
import os;
from scipy.spatial import ConvexHull as ConvH;
from PIL import Image;
def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

def get_rotation(angle):
    angle = np.radians(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
def get_translation(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
def get_scale(s):
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])
    
def apply_transform(img,A):
    if isinstance(img,list):
        w = img[0].shape[1];
        h = img[0].shape[0];
    else:
        w = img.shape[1];
        h = img.shape[0];
    coords = get_grid(w, h, True)
    x_ori, y_ori = coords[0], coords[1] 
    #get transform
    warp_coords = np.round(A@coords).astype(np.int)
    xcoord2, ycoord2 = warp_coords[0, :], warp_coords[1, :]
    #pix inside image;
    indices = np.where((xcoord2 >= 0) & (xcoord2 < w) &
                   (ycoord2 >= 0) & (ycoord2 < h))
    xpix2, ypix2 = xcoord2[indices], ycoord2[indices]
    xpix, ypix = x_ori[indices], y_ori[indices]
    #
    yy = np.round(ypix).astype(np.int);
    xx = np.round(xpix).astype(np.int);
    #color to new position
    if isinstance(img,list):
        r = [];
        for im in img:
            r.append( np.zeros_like(im) );
            r[-1][ypix2, xpix2] = im[yy, xx];
        return r;
    else:
        canvas = np.zeros_like(img)
        canvas[ypix2, xpix2] = img[yy, xx];
        return canvas;
        
def msk2t(msk):
    w = msk.shape[1];
    h = msk.shape[0];
    coords = get_grid(w, h, True);
    x_ori, y_ori = coords[0], coords[1];
    indices = np.where(msk.flatten()>0);
    xpix, ypix = x_ori[indices], y_ori[indices];
    pts = np.stack([xpix,ypix],axis=1);
    hull = ConvH(pts);
    return h//2-np.mean(ypix[hull.vertices]),w//2-np.mean(xpix[hull.vertices]);
    
def msk_center(img,msk):
    tx,ty = msk2t(msk);
    A = get_translation(tx,ty);
    r = apply_transform([img,msk],A);
    return tuple(r);

def msk_pair_center(img,smsk,tmsk):
    tx,ty = msk2t(smsk+tmsk);
    A = get_translation(tx,ty);
    r = apply_transform([img,smsk,tmsk],A);
    return tuple(r);
    
def run(**kwargs):
    opt = kwargs;
    import h5py;
    dpath = opt['data_path'];
    opath = './log/debug';
    f = h5py.File(dpath);
    img = np.array(f['img']);
    msks = np.array(f['msk']);
    imo = Image.fromarray((img*255).astype(np.uint8));
    imo.save(os.path.join(opath,'img.png'));
    for i in range(msks.shape[0]):
        msk = msks[i,...];
        msko = Image.fromarray((msk*255).astype(np.uint8),'L');
        msko.save(os.path.join(opath,'%02d_msk.png'%i));
        cimg,cmsk = msk_center(img,msk);
        cimgo = Image.fromarray((cimg*255).astype(np.uint8));
        cimgo.save(os.path.join(opath,'%02d_cimg.png'%i));
        cmsko = Image.fromarray((cmsk*255).astype(np.uint8),'L');
        cmsko.save(os.path.join(opath,'%02d_cmsk.png'%i));
    for i in range(msks.shape[0]-1):
        for j in range(i+1,msks.shape[0]):
            smsk = msks[i,...];
            tmsk = msks[j,...];
            msko = Image.fromarray(((smsk+tmsk)*255).astype(np.uint8),'L');
            msko.save(os.path.join(opath,'%02d-%02d_msk.png'%(i,j)));
            cimg,csmsk,ctmsk = msk_pair_center(img,smsk,tmsk);
            cimgo = Image.fromarray((cimg*255).astype(np.uint8));
            cimgo.save(os.path.join(opath,'%02d-%02d_cimg.png'%(i,j)));
            csmsko = Image.fromarray((csmsk*255).astype(np.uint8),'L');
            csmsko.save(os.path.join(opath,'%02d-%02d_csmsk.png'%(i,j)));
            ctmsko = Image.fromarray((ctmsk*255).astype(np.uint8),'L');
            ctmsko.save(os.path.join(opath,'%02d-%02d_ctmsk.png'%(i,j)));
        
    