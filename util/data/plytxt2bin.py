import os;
from .ply import read_ply,write_ply;

def run(**kwargs):
    dataroot = kwargs['data_path'];
    for root,ds,fs in os.walk(dataroot):
        for f in fs:
            if f.endswith('.ply'):
               fpath = os.path.join(root,f);
               print(fpath);
               try:
                    data = read_ply(fpath);
               except Exception as e:
                    print(e);
                    print(fpath);
               write_ply(fpath,points=data['points'],normal=True);