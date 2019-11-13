import json;
import os;
from ..dataset.ToyV import mv;
import numpy as np;

def get_GT(path):
    print(path)
    env = json.load(open(path));
    bA = np.array(env['box'][0]);
    dA = mv(np.mean(bA,axis=0,keepdims=True));
    bB = np.array(env['box'][1]);
    dB = mv(np.mean(bB,axis=0,keepdims=True));
    return dA[0,2],dB[0,2];

def run(**kwargs):
    path = './data/hs'
    record_path = os.path.join(path,'record')
    for f in os.listdir(record_path):
        dict = json.load(open(os.path.join(record_path,f)));
        case_path = os.path.basename(os.path.dirname(dict['A']));
        case_res_path = os.path.join(path,case_path,'res.json');
        if os.path.exists(case_res_path):
            rdict = json.load(open(case_res_path));
        else:
            gt = get_GT(os.path.join(path,case_path,'info.json'));
            rdict = {'A':0,'B':0,'dA':gt[0],'dB':gt[1]};
        rdict[dict['Res']] += 1;
        json.dump(rdict,open(case_res_path,'w'));
    return;