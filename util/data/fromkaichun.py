import numpy as np;
from numpy.linalg import inv;
import pandas as pd;
from .ply import read_ply,write_ply;
#default blender camera rotation
camrot =  np.array([[0.6859206557273865,-0.32401347160339355,0.6515582203865051],
                   [0.7276763319969177,0.305420845746994,-0.6141703724861145],
                   [0.0,0.8953956365585327,0.44527140259742737]],dtype=np.float32
                   );
                   
def rotpts(pts):
    pts = np.matmul(inv(camrot),pts.transpose(1,0));
    return pts.transpose(1,0);
    
                   
def run(**kwargs):
    data = read_ply('./log/ply/partgen/_0000/_000_partgen-h5-mini_gt.ply');
    pts = np.array(data['points']);
    pts = rotpts(pts);
    write_ply('a.ply',points=pd.DataFrame(pts));
    return;
    