import numpy as np;
import cv2 as cv;

depth = 'C:\\workspace\\PON\\log\\766fe076d4cdef8cf0117851f0671fde\\test\\Bag\\766fe076d4cdef8cf0117851f0671fde\\sp\\models\\model_normalized_r240_depth.exr0001.exr'
rgb = 'C:\\workspace\\PON\\log\\766fe076d4cdef8cf0117851f0671fde\\test\\Bag\\766fe076d4cdef8cf0117851f0671fde\\sp\\models\\model_normalized_r240.png'
dimg = cv.imread(depth,  cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) ;
rgbimg = cv.imread(rgb,cv.IMREAD_UNCHANGED) ;
print(dimg.shape);
dr = dimg[:,:,0];
dg = dimg[:,:,1];
db = dimg[:,:,2];
ddata = 0.2989 * dr + 0.5870 * dg + 0.1140 * db;
#print(ddata.shape);
#print('ddata:',np.min(ddata),np.max(ddata));
ddata = ddata.reshape((448,448));
ddata[rgbimg[:,:,3]==0] = 0.0;
normdimg = np.zeros((448,448))
normdimg = cv.normalize(ddata,normdimg,0,255.0,cv.NORM_MINMAX);
cv.imwrite(depth+'.png',normdimg.astype(np.uint8));