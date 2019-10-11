pts_num = 1200;
pts_num_gt = 1200;
grid_num = 2;
workers = 4;
optim = 'Adam';
lr = 2e-4;
nepoch = 1;
print_epoch =1;
latent_dim = 512;
mode = 'Reg';
lambda_gp = 10;
train_g = 5;
weight_decay = 0.0;
as_text = True;

from .config import parameters,NpEncoder;
import torch;

def accuracy(data,out):
    acc = {};
    L2 = ( data[4] - out['box'] )**2;
    ax = [x for x in range(1,L2.dim())];
    acc['cd'] = torch.mean(L2,dim=ax);
    reg = ( torch.sum((out['rot'])**2,dim=1) - 1.0 )**2;
    acc['reg'] = reg;
    return acc;
    
def loss(data,out):
    loss = {};
    L2 = ( data[4] - out['box'] )**2;
    ax = [x for x in range(1,L2.dim())];
    loss['cd'] = torch.mean(L2,dim=ax);
    reg = ( torch.sum((out['rot'])**2,dim=1) - 1.0 )**2;
    loss['reg'] = reg;
    loss['overall'] = torch.mean(loss['cd']) + 100.0*torch.mean(loss['reg']);
    return loss;
    
from datetime import datetime;
from util.data.ply import write_ply
import json;
import numpy as np;
import os;
from util.tools import repeat_face,write_pts2sphere;
import pandas as pd;
from PIL import Image;

bestcnt = 3;
best = np.array([10000]*bestcnt,dtype=np.float32);
bestn = [""]*bestcnt;
    
def writelog(**kwargs):
    global best;
    global bestn;
    opt = kwargs['opt'];
    iepoch = kwargs['iepoch'];
    nepoch = opt['nepoch'];
    ib = kwargs['idata'];
    nb = kwargs['ndata'] // opt['batch_size'];
    out = kwargs['out'];
    net = kwargs['net'];
    data = kwargs['data'];
    meter = kwargs['meter'];
    print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']));
    if not 'log_tmp' in opt.keys():
        opt['log_tmp'] = opt['log']+os.sep+opt['net']+'_'+opt['config']+'_'+opt['mode']+'_'+str(datetime.now()).replace(' ','-').replace(':','-');
        os.mkdir(opt['log_tmp']);
        with open(opt['log_tmp']+os.sep+'options.json','w') as f:
            json.dump(opt,f,cls=NpEncoder);
        nparam = 0;
        with open(opt['log_tmp']+os.sep+'net.txt','w') as logtxt:
            print(str(kwargs['net']),file=logtxt);
            for p in parameters(kwargs['net']):
                nparam += torch.numel(p);
            print('nparam:%d'%nparam,file=logtxt);
        
    with open(opt['log_tmp']+os.sep+'log.txt','a') as logtxt:
        print('['+str(datetime.now())+'][%d/%d,%d/%d]'%(iepoch,nepoch,ib,nb)+'training:'+str(kwargs['istraining']),file=logtxt);
        print(json.dumps(meter,cls=NpEncoder),file=logtxt);
        
    if not kwargs['istraining'] and ib >= nb-1:
        if meter['cd'].overall_meter.avg < best[-1]:
            fn = bestn[-1];
            if fn:
                os.remove(opt['log_tmp']+os.sep+fn);
            fn = 'net_'+str(datetime.now()).replace(' ','-').replace(':','-')+'.pth';
            best[-1] = meter['cd'].overall_meter.avg;
            bestn[-1] = fn;
            torch.save(net.state_dict(),opt['log_tmp']+os.sep+fn);
            idx = np.argsort(best);
            best = best[idx];
            bestn = [bestn[x] for x in idx.tolist()];
            bestdict = dict(zip(bestn, best.tolist()));
            print(best);
            print(bestn);
            print(bestdict);
            with open(opt['log_tmp']+os.sep+'best.json','w') as f:
                json.dump(bestdict,f);

    if opt['ply'] and not kwargs['istraining']:
        ply_path = opt['log_tmp']+os.sep+'ply';
        if not os.path.exists(ply_path):
            os.mkdir(ply_path);
        x = out['grid_x'];
        x = x.data.cpu().numpy();
        y = out['y'];
        yout = y.data.cpu().numpy();
        ysrc = data[3];
        ysrc = ysrc.data.cpu().numpy();
        ytgt = data[4];
        ytgt = ytgt.data.cpu().numpy();
        yall = data[5];
        yall = yall.data.cpu().numpy();
        cat = data[-1];
        im = data[0];
        im = im.data.cpu().numpy();
        src = data[1];
        src = src.data.cpu().numpy();
        tgt = data[2];
        tgt = tgt.data.cpu().numpy();
        for i in range(y.shape[0]):
            fidx = repeat_face(x[i,...],opt['grid_num'],8);
            T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
            face = np.zeros(shape=[fidx.shape[0]],dtype=T);
            for fi in range(fidx.shape[0]):
                face[fi] = (3,fidx[fi,0],fidx[fi,1],fidx[fi,2]);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_all.ply'%(ib,i,cat[i]),points = pd.DataFrame(yall[i,...]),faces=pd.DataFrame(face),as_text=opt['as_text']);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_src.ply'%(ib,i,cat[i]),points = pd.DataFrame(ysrc[i,...]),faces=pd.DataFrame(face[0:12]),as_text=opt['as_text']);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_tgt.ply'%(ib,i,cat[i]),points = pd.DataFrame(ytgt[i,...]),faces=pd.DataFrame(face[0:12]),as_text=opt['as_text']);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_out.ply'%(ib,i,cat[i]),points = pd.DataFrame(yout[i,...]),faces=pd.DataFrame(face[0:12]),as_text=opt['as_text']);
            write_ply(ply_path+os.sep+'_%04d_%03d_%s_oall.ply'%(ib,i,cat[i]),points = pd.DataFrame(np.concatenate([ysrc[i,...],yout[i,...]],axis=0)),faces=pd.DataFrame(face),as_text=opt['as_text']);
            img = im[i,...];
            img = img.transpose((1,2,0));
            img = Image.fromarray(np.uint8(255.0*img));
            img.save(ply_path+os.sep+'_%04d_%03d_%s_input.png'%(ib,i,cat[i]));
            img = src[i,...];
            img = Image.fromarray(np.uint8(255.0*img));
            img.save(ply_path+os.sep+'_%04d_%03d_%s_src.png'%(ib,i,cat[i]));
            img = tgt[i,...];
            img = Image.fromarray(np.uint8(255.0*img));
            img.save(ply_path+os.sep+'_%04d_%03d_%s_tgt.png'%(ib,i,cat[i]));

