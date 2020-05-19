import torch;
import traceback
import importlib
from torch.utils.data import DataLoader;
from torch import optim;
from torch.optim import lr_scheduler;
from .tools import *;
import json;
from config.config import NpEncoder;
from chamferdist import ChamferDistance;
distChamfer =  ChamferDistance();

def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad = True;
            
def prepare_data(data):
                
                
    return [im,dm]
            

def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        for k in kwargs.keys():
            if not kwargs[k] is None:
                opt[k] = kwargs[k];
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
        if torch.cuda.is_available():
            net = net.cuda();
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    try:
        m = importlib.import_module('util.dataset.'+opt['dataset']);
        train_data = m.Data(opt,True);
        val_data = m.Data(opt,False);
        train_load = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    
    #run the code
    optimizer = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);
    sheduler = lr_scheduler.ExponentialLR(optimizer,0.9);
    #load pre-trained
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
    
    for iepoch in range(opt['nepoch']):
        #
        net.train();
        train_meters = {};
        for i, data in enumerate(train_load,0):
            optimizer.zero_grad();
            data2cuda(data);
            din = prepare_data(data);
            out = net(*din);
            loss = config.loss(data,out);
            loss['overall'].backward();
            optimizer.step();
            config.writelog(net=net,data=data,out=out,meter=train_meters,opt=opt,iepoch=iepoch,idata=i,ndata=len(train_data),optim=optimizer,istraining=True);
        if iepoch % opt['lr_decay_freq'] == 0: 
            sheduler.step();