import zipfile as zpf;
import os;
dataroot = '/cephfs/siyu/cage'
pndata = 'partnet.zip';
spndata = 'shapenet.zip';

if __name__ == '__main__':
    with zpf.ZipFile(os.path.join(dataroot,pndata),'r') as pnzip:
        with zpf.ZipFile(os.path.join(dataroot,spndata),'r') as spnzip:
            for info in spnzip.infolist():
                info.is_dir();
               
    