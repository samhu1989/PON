import numpy as np;
from numpy.linalg import inv;
import pandas as pd;
from .ply import read_ply,write_ply;
#default blender camera rotation
camrot =  np.array([[0.6859206557273865,-0.32401347160339355,0.6515582203865051],
                   [0.7276763319969177,0.305420845746994,-0.6141703724861145],
                   [0.0,0.8953956365585327,0.44527140259742737]],dtype=np.float32
                   );
                   
mva = np.array(
        [
            [-0.0000,  1.0000,  0.0000,  0.0000],
            [-0.0000, -0.0000,  1.0000,  0.0000],
            [ 1.0000,  0.0000,  0.0000, -2.0000],
            [-0.0000, -0.0000, -0.0000,  1.0000]
        ],dtype=np.float32
        
)            
                   
mvb = np.array( 
    [
        [1,0.7277,0.0000,-0.3960],
        [-0.3240,0.3054,0.8954,-0.3731],
        [0.6516,-0.6142,0.4453,-11.0],
        [-0.0000,0.0000,-0.0000,1.0000]
    ],dtype=np.float32
    );
    
    
                   
def rotpts(pts):
    pts = pts.transpose(1,0);
    pts = np.concatenate([pts,np.ones([1,pts.shape[1]],dtype=np.float32)],axis=0);
    pts = np.matmul(mva,pts);
    pts = np.matmul(inv(mvb),pts);
    pts = pts.transpose(1,0)
    return pts[:,:3];
    
def run(**kwargs):
    data = read_ply('./data/chair.ply');
    pts = np.array(data['points']);
    pts = rotpts(pts);
    write_ply('ch.ply',points=pd.DataFrame(pts));
    return;
    