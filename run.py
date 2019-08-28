from __future__ import print_function
from __future__ import division
#python import
import sys;
import traceback
import importlib
import os;
#project import
sys.path.append('./');
from util.options import get_opt
from util.options import usage;
import torch;

if __name__ == "__main__":
    #get options from command line inputs
    torch.backends.cudnn.enabled = False;
    opt = get_opt();
    try:
        m = importlib.import_module(opt.execute)
        m.run(**opt.__dict__);
    except Exception as e:
        usage();
        print(e);
        traceback.print_exc();
