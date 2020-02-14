import zipfile as zpf;
import os;
import numpy as np;
import sys;
import process_job as pj;

dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';

def extract_job_data(job):
    id = job.split('/')[2];
    #create tmp workspace
    workpath = os.path.join(dataroot,'tmp',id);
    if not os.path.exists(workpath):
        os.makedirs(workpath);
    print(job);
    with zpf.ZipFile(os.path.join(dataroot,spndata),'r') as spnzip:
        for name in spnzip.namelist():
            if name.startswith(job):
                spnzip.extract(name,path=workpath);
    with zpf.ZipFile(os.path.join(dataroot,pndata),'r') as pnzip:
        partdata = 'partnet/'+id+'/'
        for name in pnzip.namelist():
            if name.startswith(partdata):
                pnzip.extract(name,path=workpath);

if __name__ == '__main__':
    lst = sys.argv[1];
    with open(os.path.join(dataroot,lst) ,'r') as joblst:
        for job in joblst:
            job = job.rstrip('\n');
            extract_job_data(job);
            pj.do_one(job);
            