import zipfile as zpf;
import os;
import numpy as np;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';
caseperjob = 80;
cat_lst = [];
if __name__ == '__main__':
    with zpf.ZipFile(os.path.join(dataroot,spndata),'r') as spnzip:
        casecnt = 0;
        jobcnt = 0;
        current_lst = None;
        newlst = True;
        debug_lst = open(os.path.join(dataroot,'job_debug.txt'),'w');
        for info in spnzip.infolist():
            if casecnt == 0 and newlst:
                if current_lst is not None:
                    current_lst.flush();
                    current_lst.close();
                current_lst = open(os.path.join(dataroot,'job_%03d.txt'%jobcnt),'w');
                newlst = False;
                print(jobcnt);
                jobcnt += 1;
            if len(info.filename.split(os.sep)) == 4:
                print(info.filename,file = current_lst );
                casecnt += 1;
                if casecnt >= caseperjob:
                    casecnt = 0;
                    newlst = True;
                cat = info.filename.split(os.sep)[1];
                if not cat in cat_lst:
                    cat_lst.append(cat);
                    print(info.filename,file = debug_lst);
               
    