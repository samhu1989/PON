import torch;
import traceback
import importlib
from torch.utils.data import DataLoader;

def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        opt.update(kwargs);
    except Exception as e:
        print(e);
        traceback.print_exc();
    #get network
    try:
        m = importlib.import_module(opt['net']);
        net = m.Net(**kwargs);
    except Exception as e:
        print(e);
        traceback.print_exc();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+kwargs['dataset']);
        train_data = m.Data(kwargs,True);
        val_data = m.Data(kwargs,False);
        train_load = DataLoader(train_data,batch_size=kwargs['batch_size'],shuffle=True,num_workers=kwargs['workers']);
        val_load = DataLoader(val_data,batch_size=kwargs['batch_size'],shuffle=False,num_workers=kwargs['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
    net.eval();
    #validation
    for data in val
    
    