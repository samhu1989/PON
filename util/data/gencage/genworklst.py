import zipfile as zpf;
import os;
import numpy as np;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';
caseperjob = 80;

if __name__ == '__main__':
    with zpf.ZipFile(os.path.join(dataroot,spndata),'r') as spnzip:
        casecnt = 0;
        jobcnt = 0;
        current_lst = None;
        for info in spnzip.infolist():
            if casecnt == 0:
                if current_lst is not None:
                    current_lst.close();
                current_lst = open(os.path.join(dataroot,'job_%02d.txt'%jobcnt),'w');
                jobcnt += 1;
                print(jobcnt);
            if info.is_dir() and len(info.filename.split(os.sep)) == 2:
                print(info.filename,file = current_lst );
                casecnt += 1;
                if casecnt >= caseperjob:
                    casecnt = 0;
               
    