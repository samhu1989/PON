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
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
    except Exception as e:
        print(e);
        traceback.print_exc();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+opt['dataset']);
        train_data = m.Data(opt,True);
        val_data = m.Data(opt,False);
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
    #
    net.eval();
    #validation
    for data in val_load:
        out = net(data);
        acc = config.accuracy(out,data);
        cat
        
        
        
    net.train();
    #train
    for data in train_data:
        out = net(data);
        loss = config.loss(out,data);
        
        
    
    