from __future__ import print_function
from __future__ import division
#python import
import sys;
import traceback
import importlib
import os;
#project import
sys.path.append('./');
sys.path.append('./ext/cd/install/Lib/site-packages');
sys.path.append('./ext/cd/install/lib/python%s/site-packages'%(sys.version[:3]));
print(sys.path)
from util.options import get_opt
from util.options import usage;

if __name__ == "__main__":
    #get options from command line inputs
    opt = get_opt();
    try:
        m = importlib.import_module(opt.execute)
        m.run(**opt.__dict__);
    except Exception as e:
        usage();
        print(e);
        traceback.print_exc();
