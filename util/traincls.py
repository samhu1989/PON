import torch;
import traceback
import importlib
from torch.utils.data import DataLoader;
from torch import optim;
from torch.optim import lr_scheduler;
from .tools import *;
import json;
from config.config import NpEncoder;
from datetime import datetime;
import torch;
import torch.nn as nn;
def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            if isinstance(data[i],torch.FloatTensor):
                data[i].requires_grad = True;

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
        train_load = DataLoader(train_data,batch_size=opt['batch_size'],shuffle=True,num_workers=opt['workers']);
        val_load = DataLoader(val_data,batch_size=opt['batch_size'],shuffle=False,num_workers=opt['workers']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['config']+'_'+opt['mode']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
        os.mkdir(opt['log_tmp']);
        with open(opt['log_tmp']+os.sep+'options.json','w') as f:
            json.dump(opt,f,cls=NpEncoder);
        nparam = 0;
        with open(opt['log_tmp']+os.sep+'net.txt','w') as logtxt:
            print(str(net),file=logtxt);
            for p in net.parameters():
                nparam += torch.numel(p);
            print('nparam:%d'%nparam,file=logtxt);
    #run the code
    optimizer = eval('optim.'+opt['optim'])(net.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay']);
    sheduler = lr_scheduler.ExponentialLR(optimizer,0.9);
    #load pre-trained
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
    #
    for iepoch in range(opt['nepoch']):
        net.eval();
        #validation
        correct = 0.0
        total = 0.0
        #
        for i, data in enumerate(val_load,0):
            data2cuda(data);
            labels = data[1];
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += float(labels.size(0));
            correct += float((predicted.cpu() == labels.cpu()).sum().item());
        print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * float(correct) / float(total)))
                
        torch.save(net.state_dict(),opt['log_tmp']+os.sep+'latest_%d.pth'%iepoch);
        torch.save(optimizer.state_dict(),opt['log_tmp']+os.sep+'latest_opt_%d.pth'%iepoch);
        if iepoch > 0:
            os.remove(opt['log_tmp']+os.sep+'latest_%d.pth'%(iepoch-1));
            os.remove(opt['log_tmp']+os.sep+'latest_opt_%d.pth'%(iepoch-1));
        #
        net.train();
        running_loss = 0.0;
        criterion = nn.CrossEntropyLoss();
        for i, data in enumerate(train_load,0):
            optimizer.zero_grad();
            data2cuda(data);
            out = net(data);
            loss = criterion(out,data[1]);
            loss.backward();
            optimizer.step();
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d ,%5d/%d] loss: %.3f' %(iepoch + 1,i + 1,len(train_data)//opt['batch_size'],running_loss / float(i+1)))
            
        if iepoch % opt['lr_decay_freq'] == 0: 
            sheduler.step();