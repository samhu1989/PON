import zipfile as zpf;
import os;
import numpy as np;
import sys;
dataroot = '/cephfs/siyu/cage';
pndata = 'partnet.zip';
spndata = 'shapenet.zip';

def extract_job_data(job):
    print(job);

    

if __name__ == '__main__':
    lst = sys.argv[1];
    with open(os.path.join(dataroot,lst) ,'r') as joblst:
        for job in joblst:
            extract_job_data(job);
            exit();
            