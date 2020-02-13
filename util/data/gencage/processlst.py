import zipfile as zpf;
import os;
import numpy as np;
import sys;

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
        spnzip.extract(job,path=workpath);
    with zpf.ZipFile(os.path.join(dataroot,pndata),'r') as pnzip:
        pnzip.extract('partnet/'+id,path=workpath);
    
if __name__ == '__main__':
    lst = sys.argv[1];
    with open(os.path.join(dataroot,lst) ,'r') as joblst:
        for job in joblst:
            extract_job_data(job);
            exit();
            