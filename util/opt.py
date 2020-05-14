import torch;
import traceback
import importlib
from torch.utils.data import DataLoader;
from torch import optim;
from torch.optim import lr_scheduler;
from .tools import *;
import json;
from config.config import NpEncoder;
from .sample import randsphere2,triangulateSphere;
from PIL import Image;
from util.data.ply import write_ply;
import pandas as pd;
from chamferdist import ChamferDistance;
distChamfer =  ChamferDistance();

def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
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
    
    #run the code
    
    #load pre-trained
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
    #
    grid = randsphere2(opt['pts_num']);
    grid = torch.from_numpy(grid.reshape(1,grid.shape[0],grid.shape[1]).astype(np.float32)).cuda();
    grid = grid.repeat(opt['batch_size'],1,1);
    hull = triangulateSphere(grid.cpu().data.numpy());
    fidx = hull[0].simplices;
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    
    #
    done_cat = [];
    for i, data in enumerate(val_load,0):
        if data[3][0] in done_cat:
            continue;
        else:
            done_cat.append(data[3][0]);
        net.eval();
        with torch.no_grad():
            data2cuda(data);
            out = net(data[0],grid);
        img = data[0].data.cpu().numpy();
        ygt = data[1].data.cpu().numpy();
        yout = out['y'].data.cpu().numpy();
        img = img.transpose((0,2,3,1));
        for j in range(opt['batch_size']):
            cat = data[3][j]
            opath = './log/debug/'+cat;
            if not os.path.exists(opath):
                os.mkdir(opath);
            Image.fromarray((img[j,:,:]*255).astype(np.uint8)).save(opath+'/%s.png'%(data[4][j]));
            write_ply(opath+'/%s.ply'%(data[4][j]),points=pd.DataFrame(yout[j,:,:]),faces=pd.DataFrame(face));
            write_pts2sphere(opath+'/%s_gt.ply'%(data[4][j]),ygt[j,:,:]);
        
        X = torch.FloatTensor(opt['batch_size'],3,224,224);
        X.data.normal_(0,0.5);
        X.requires_grad = True;
        optimizer = eval('optim.'+opt['optim'])([X],lr=opt['lr'],weight_decay=opt['weight_decay']);

        for iter in range(opt['nepoch']):
            optimizer.zero_grad();
            out = net(torch.sigmoid(X.cuda()));
            dist1, dist2, _, _ = distChamfer(out['y'],data[1]);
            loss = torch.mean(dist1) + torch.mean(dist2);
            loss.backward();
            print(iter,':',loss.item());
            if loss.item() < 0.001:
                break;
            optimizer.step();
        
        ximg = torch.sigmoid(X).data.numpy();        
        ximg = ximg.transpose((0,2,3,1));
        for j in range(opt['batch_size']):
            cat = data[3][j]
            opath = './log/debug/'+cat;
            if not os.path.exists(opath):
                os.mkdir(opath);
            Image.fromarray((ximg[j,:,:]*255).astype(np.uint8)).save(opath+'/%s_opt.png'%(data[4][j]));
            
        with torch.no_grad():
            data2cuda(data);
            out = net(torch.sigmoid(X.cuda()),grid);
        yout = out['y'].data.cpu().numpy();
        for j in range(opt['batch_size']):
            cat = data[3][j]
            opath = './log/debug/'+cat;
            write_ply(opath+'/%s_opt.ply'%(data[4][j]),points=pd.DataFrame(yout[j,:,:]),faces=pd.DataFrame(face));
            
            
    