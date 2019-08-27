import torch;
import traceback
import importlib
from torch.utils.data import DataLoader;
from torch import optim;
from .tools import *;

def run(**kwargs):
    #get configuration
    try:
        config = importlib.import_module('config.'+kwargs['config']);
        opt = config.__dict__;
        opt.update(kwargs);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get network
    try:
        m = importlib.import_module('net.'+opt['net']);
        net = m.Net(**opt);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
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
        exit();
    #run the code
    optimizer = eval('optim.'+opt['optim'])(config.parameters(net),lr=opt['lr'],weight_decay=opt['weight_decay']);
    for iepoch in range(opt['nepoch']):
        net.eval();
        #validation
        val_meters = {};
        for i, data in enumerate(val_load,0):
            with torch.no_grad():
                out = net(data);
            acc = config.accuracy(data,out);
            for k,v in acc:
                if k in val_meters.keys():
                    val_meters[k].update(v,data[-1]);
                else:
                    val_meters[k] = AvgMeterGroup(k);
                    val_meters[k].update(v,data[-1]);
            config.writelog(data=data,out=out,meter=val_meters,opt=opt,iepoch=iepoch,idata=i,ndata=len(val_data),istraining=False);
        
        net.train();
        #train
        train_meters = {};
        for i, data in enumerate(train_load,0):
            optimizer.zero_grad();
            out = net(data);
            loss = config.loss(data,out);
            for k,v in loss:
                if k == 'overall':
                    continue;
                if k in train_meters.keys():
                    train_meters[k].update(v,data[-1]);
                else:
                    train_meters[k] = AvgMeterGroup(k);
                    train_meters[k].update(v,data[-1]);
            loss['overall'].backward();
            optimizer.step();
            config.writelog(data=data,out=out,meter=train_meters,opt=opt,iepoch=iepoch,idata=i,ndata=len(train_data),istraining=True);