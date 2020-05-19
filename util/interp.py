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
        val_load = DataLoader(val_data,batch_size=opt['batch_size']*2,shuffle=False,num_workers=opt['workers']);
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
    ws = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0];
    done_cat = {};
    for i, data in enumerate(val_load,0):
        if data[3][0] in done_cat.keys():
            continue;
        else:
            done_cat[data[3][0]]=data;
        img = data[0].data.cpu().numpy();
        img = img.transpose((0,2,3,1));
        ygt = data[1].data.cpu().numpy();
        for j in range(opt['batch_size']*2):
                cat = data[3][j];
                opath = './log/interp/'+cat;
                if not os.path.exists(opath):
                    os.mkdir(opath);
                Image.fromarray((img[j,...]*255).astype(np.uint8)).save(opath+'/%s.png'%(data[4][j]));
                write_pts2sphere(opath+'/%s_gt.ply'%(data[4][j]),ygt[j,:,:]);
                
        net.eval();
        for w in ws:
            with torch.no_grad():
                data2cuda(data);
                out = net.interp(data[0][0:opt['batch_size'],...],data[0][opt['batch_size']:2*opt['batch_size'],...],w,grid);
        
            yout = out['y'].data.cpu().numpy();
            img = img.transpose((0,2,3,1));
            for j in range(opt['batch_size']):
                cat = data[3][j]
                opath = './log/interp/'+cat;
                if not os.path.exists(opath):
                    os.mkdir(opath);
                write_ply(opath+'/%s_%s_%f.ply'%(data[4][j],data[4][j+opt['batch_size']],w),points=pd.DataFrame(yout[j,:,:]),faces=pd.DataFrame(face));
    klst = list(done_cat.keys());
    for ni,ci in enumerate(klst):
        datai = done_cat[ci];
        for nj in range(ni+1,len(klst)):
            cj = klst[nj];
            dataj = done_cat[cj];
            imgi = datai[0].data.cpu().numpy();
            imgi = imgi.transpose((0,2,3,1));
            ygti = datai[1].data.cpu().numpy();
            #
            imgj = dataj[0].data.cpu().numpy();
            imgj = imgj.transpose((0,2,3,1));
            ygtj = dataj[1].data.cpu().numpy();
            opath = './log/interp/'+ci+'_'+cj;
            #
            if not os.path.exists(opath):
                os.mkdir(opath);
            for j in range(opt['batch_size']):
                Image.fromarray((imgi[j,...]*255).astype(np.uint8)).save(opath+'/%s.png'%(datai[4][j]));
                write_pts2sphere(opath+'/%s_gt.ply'%(datai[4][j]),ygti[j,:,:]);
                Image.fromarray((imgj[j,...]*255).astype(np.uint8)).save(opath+'/%s.png'%(dataj[4][j]));
                write_pts2sphere(opath+'/%s_gt.ply'%(dataj[4][j]),ygtj[j,:,:]);
            #
            net.eval();
            for w in ws:
                with torch.no_grad():
                    data2cuda(data);
                    out = net.interp(datai[0][0:opt['batch_size'],...],dataj[0][0:opt['batch_size'],...],w,grid);
                yout = out['y'].data.cpu().numpy();
                for j in range(opt['batch_size']):
                    write_ply(opath+'/%s_%s_%f.ply'%(datai[4][j],dataj[4][j],w),points=pd.DataFrame(yout[j,:,:]),faces=pd.DataFrame(face));
            
            
        
        
        