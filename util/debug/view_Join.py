import traceback
import importlib
import sys;
import torch;
import numpy as np;
from torch import optim;
from matplotlib import pyplot as plt
from matplotlib import animation;
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import axes3d as p3;
from functools import partial;
from torch.utils.data import DataLoader;
from ..data.gen_toybox import box_face;
from net.g.box import Box;
from util.dataset.ToyV import recon;
import os;
from PIL import Image;
from util.data.ply import write_ply;
import pandas as pd;
from util.tools import partial_restore;
import h5py;
import heapq;

red_box = np.array(
    [
     [255,0,0],[255,0,0],[255,0,0],[255,0,0],
     [255,0,0],[255,0,0],[255,0,0],[255,0,0]
    ],dtype=np.uint8);
blue_box = np.array(
    [
     [0,0,255],[0,0,255],[0,0,255],[0,0,255],
     [0,0,255],[0,0,255],[0,0,255],[0,0,255]
    ],dtype=np.uint8);


def data2cuda(data):
    for i in range(len(data)):
        if torch.cuda.is_available() and isinstance(data[i],torch.Tensor):
            data[i] = data[i].cuda();
            data[i].requires_grad_(True);
            
def parsegt(vec):
    coord = np.array([[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]],dtype=np.float32);
    ss = vec[:3];
    coords = ss[np.newaxis,:]*coord;
    center = vec[3:6];
    srot = vec[6:15];
    box = np.dot(coords,srot.reshape(3,3)) + center[np.newaxis,:];
    return  box,center[np.newaxis,:];
    
def writebox(path,box,colors=None):
    fidx = box_face;
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    bn = len(box);
    face = np.zeros(shape=[bn*fidx.shape[0]],dtype=T);
    for i in range(bn*fidx.shape[0]):
        nn = i // fidx.shape[0];
        ni = i % fidx.shape[0];
        face[i] = (3,fidx[ni,0]+nn*8,fidx[ni,1]+nn*8,fidx[ni,2]+nn*8);
    pts = np.concatenate(box,axis=0);
    if colors is None:
        write_ply(path,points=pd.DataFrame(pts.astype(np.float32)),faces=pd.DataFrame(face));
    else:
        colors = np.concatenate(colors,axis=0);
        pointsc = pd.concat([pd.DataFrame(pts.astype(np.float32)),pd.DataFrame(colors)],axis=1,ignore_index=True);
        write_ply(path,points=pointsc,faces=pd.DataFrame(face),color=True);
    

def writegt(path,boxgt):
    writebox(os.path.join(path,'_002_000_gt.ply'),boxgt);
    
def writeout(path,box,color,msk):
    for i in range(1,len(box)):
        bout = box[0:i];
        cout = color[0:i];
        writebox(os.path.join(path,'_002_%03d_out.ply'%i),bout,cout);
        im = Image.fromarray((msk[i]*255).astype(np.uint8),mode='L');
        im.save(os.path.join(path,'_001_%03d_msk.png'%i));

def run(**kwargs):
    global iternum;
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
    print(opt['mode']);
    #get network
    try:
        m = importlib.import_module('net.'+opt['touch_net']);
        touchnet = m.Net(**opt);
        #
        m = importlib.import_module('net.'+opt['box_net']);
        boxnet = m.Net(**opt);
        #
        m = importlib.import_module('net.'+opt['touchpt_net']);
        touchptnet = m.Net(**opt);
        #
        if torch.cuda.is_available():
            touchnet = touchnet.cuda();
            touchnet.eval();
            boxnet = boxnet.cuda();
            boxnet.eval();
            touchptnet = touchptnet.cuda();
            touchptnet.eval();
        #
        partial_restore(touchnet,opt['touch_model']);
        partial_restore(boxnet,opt['box_model']);
        partial_restore(touchptnet,opt['touchpt_model']);
    except Exception as e:
        print(e);
        traceback.print_exc();
        exit();
    #get dataset
    dpath = os.path.join(opt['data_path'],'test')
    paths = os.listdir(dpath);
    opath = './log/join'
    if not os.path.exists(opath):
        os.makedirs(opath);
    cat_lst = ['Chair','Table']
    for cat in cat_lst:
        cpath = os.path.join(opt['data_path'],'test',cat);
        copath = os.path.join(opath,cat);
        if not os.path.exists(copath):
            os.mkdir(copath);
        slst = os.listdir(cpath);
        for f in slst:
            id = os.path.basename(f).split('.')[-2];
            fopath = os.path.join(copath,'_'+id);
            if not os.path.exists(fopath):
                os.mkdir(fopath);
            h5f = h5py.File(os.path.join(cpath,f),'r');
            img = np.array(h5f['img']);
            msk = np.array(h5f['msk']);
            smsk = np.array(h5f['smsk']);
            box = np.array(h5f['box']);
            num = box.shape[0];
            bdata = [];
            img_lst = [];
            smsk_lst = [];
            tmsk_lst = [];
            box_lst = [];
            #for each part
            rate = opt['user_rate'];
            for i in range(num):
                msk_rate = ( np.sum(msk[i,...]) / np.sum(smsk[i,...]) );
                if msk_rate > rate:
                    img_lst.append(img);
                    smsk_lst.append(msk[i,...]);
                    tmsk_lst.append(msk[i,...]);
                    box_lst.append(box[i,...]);
            #
            img = np.stack(img_lst,axis=0);
            smsk = np.stack(smsk_lst,axis=0);
            tmsk = np.stack(tmsk_lst,axis=0);
            bdata.append(torch.from_numpy(img).cuda());
            bdata.append(torch.from_numpy(smsk).cuda());
            bdata.append(torch.from_numpy(tmsk).cuda());
            with torch.no_grad():
                boxout = boxnet(bdata);
            size = np.prod(boxout['ss'].data.cpu().numpy(),axis=1);
            undone_queue = [];
            for idx,v in enumerate(size):
                heapq.heappush(undone_queue,(-v,idx));
            done_queue = [];
            msk_in = [];
            box_out = [];
            box_color = [];
            box_gt = [];
            while len(undone_queue) > 0:
                if len(done_queue) > 0 :
                    itop = heapq.heappop(done_queue);
                else:
                    itop = heapq.heappop(undone_queue);
                    ci = itop[1];
                    bo = boxout['sb'].data.cpu().numpy()[ci,...];
                    bgt,t = parsegt(box_lst[ci]);
                    box_out.append(bo+t);
                    box_gt.append(bgt)
                    box_color.append(red_box);
                    msk_in.append(bdata[1].data.cpu().numpy()[ci,...]);
                ci = itop[1];
                tdata = [];
                tptdata = [];
                unfinish = [];
                while len(undone_queue) > 0:
                    jtop = heapq.heappop(undone_queue);
                    cj = jtop[1];
                    tdata.append(  bdata[0][0,...].unsqueeze(0) );
                    tdata.append( (bdata[1][ci,...]).unsqueeze(0) );
                    tdata.append( (bdata[1][cj,...]).unsqueeze(0) );
                    with torch.no_grad():
                        touchout = touchnet(tdata);
                    if touchout['y'].data.cpu().numpy()[0][0] > 0.5:
                        tptdata.append(tdata[0]);
                        tptdata.append(tdata[1]);
                        tptdata.append(tdata[2]);
                        tptdata.append(None);
                        vec = np.zeros([1,21],dtype=np.float32);
                        vec[0,:3] = boxout['ss'].data.cpu().numpy()[ci,...];
                        vec[0,3:6] = boxout['sr1'].data.cpu().numpy()[ci,...];
                        vec[0,6:9] = boxout['sr2'].data.cpu().numpy()[ci,...];
                        vec[0,9:12] = boxout['ss'].data.cpu().numpy()[cj,...];
                        vec[0,15:18] = boxout['sr1'].data.cpu().numpy()[cj,...];
                        vec[0,18:21] = boxout['sr2'].data.cpu().numpy()[cj,...];
                        tptdata.append(torch.from_numpy(vec).cuda());
                        with torch.no_grad():
                            tptout = touchptnet(tptdata);
                        heapq.heappush(done_queue,jtop);
                        bo = boxout['sb'].data.cpu().numpy()[cj,...];
                        bgt,_ = parsegt(box_lst[cj]);
                        t = tptout['t'].data.cpu().numpy();
                        box_out.append(bo+t);
                        box_gt.append(bgt);
                        box_color.append(blue_box);
                        msk_in.append(bdata[1].data.cpu().numpy()[cj,...]);
                    else:
                        unfinish.append(jtop);
                for p in unfinish:
                    heapq.heappush(undone_queue,p);
            im = Image.fromarray((bdata[0][0,...].data.cpu().numpy()*255).astype(np.uint8));
            im.save(os.path.join(fopath,'_002_000_im.png'));
            writegt(fopath,box_gt);
            writeout(fopath,box_out,box_color,msk_in);
            exit();
            
        
    '''
    print('bs:',opt['batch_size']);
        
    if opt['model']!='':
        partial_restore(net,opt['model']);
        print("Previous weights loaded");
           
    if opt['model']!='':
        outdir = os.path.dirname(opt['model'])+os.sep+'view';
        if not os.path.exists(outdir):
            os.mkdir(outdir);
            
    #run the code
    tri = box_face;
    fidx = tri;
    
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    face = np.zeros(shape=[2*fidx.shape[0]],dtype=T);
    for i in range(2*fidx.shape[0]):
        if i < fidx.shape[0]:
            face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
        else:
            face[i] = (3,fidx[i-fidx.shape[0],0]+8,fidx[i-fidx.shape[0],1]+8,fidx[i-fidx.shape[0],2]+8);
    '''