import h5py;
from PIL import Image;
import os;
import numpy as np;

def run(**kwargs):
    h5 = h5py.File(kwargs['data_path']);
    imgo = Image.fromarray( (np.array(h5['img'])*255).astype(np.uint8) );
    imgo.save('./log/img.png');
    msks = np.array(h5['msk']);
    smsk = msks[0,...];
    tmsk = msks[3,...];
    to = Image.fromarray((tmsk*255).astype(np.uint8),'L');
    so = Image.fromarray((smsk*255).astype(np.uint8),'L');
    tso = Image.fromarray(((smsk+tmsk)*255).astype(np.uint8),'L');
    so.save('./log/smsk.png');
    to.save('./log/tmsk.png');
    tso.save('./log/stmsko.png');
    
    