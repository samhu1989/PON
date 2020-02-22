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
from net.cageutil import rot9np,normalize,rot6d,rot9;
import torch.nn.functional as F;
from util.loss.bcd import box_cd_batch;
from scipy.spatial.transform import Rotation as R

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
    vec = vec.copy();
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
    for i in range(1,len(box)+1):
        bout = box[0:i];
        cout = color[0:i];
        writebox(os.path.join(path,'_002_%03d_out.ply'%i),bout,cout);
        im = Image.fromarray((msk[i-1]*255).astype(np.uint8),mode='L');
        im.save(os.path.join(path,'_001_%03d_msk.png'%i));
        
def add_gt_for_eval(vec,s_gt,t_gt,r1_gt,r2_gt):
    vec = vec.astype(np.float32);
    s_gt.append(torch.from_numpy(vec[:3]).cuda());
    t_gt.append(torch.from_numpy(vec[3:6][np.newaxis,:]).cuda());
    r1_gt.append(torch.from_numpy(vec[6:9]).cuda());
    r2_gt.append(torch.from_numpy(vec[9:12]).cuda());

def add_out_for_eval(s,t,r1,r2,s_out,t_out,r1_out,r2_out):
    s_out.append(s);
    t_out.append(t);
    r1_out.append(r1);
    r2_out.append(r2);
    
def eval(s_out,t_out,r1_out,r2_out,s_gt,t_gt,r1_gt,r2_gt):
    so = torch.stack(s_out,dim=0);
    to = torch.stack(t_out,dim=0);
    r1o = torch.stack(r1_out,dim=0);
    r2o = torch.stack(r2_out,dim=0);
    sgt = torch.stack(s_gt,dim=0);
    tgt = torch.stack(t_gt,dim=0);
    r1gt = torch.stack(r1_gt,dim=0);
    r2gt = torch.stack(r2_gt,dim=0);
    with torch.no_grad():
        return box_cd_batch(so,to,r1o,r2o,sgt,tgt,r1gt,r2gt);

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
    opath = './log/join'
    if not os.path.exists(opath):
        os.makedirs(opath);
    cat_lst = os.listdir(dpath);
    #cat_lst = ['Chair','Table'];
    eval_log = open('./log/join/'+opt['mode']+'_eval.csv','w');
    eval_all_sum = 0.0;
    eval_all_cnt = 0.0;
    for cat in cat_lst:
        eval_cat_sum = 0.0;
        eval_cat_cnt = 0.0;
        cpath = os.path.join(opt['data_path'],'test',cat);
        copath = os.path.join(opath,'new_'+cat+'_'+opt['mode']);
        if not os.path.exists(copath):
            os.mkdir(copath);
        slst = os.listdir(cpath);
        for f in slst:
            id = os.path.basename(f).split('.')[-2];
            fopath = os.path.join(copath,'_'+id);
            if not os.path.exists(fopath):
                os.mkdir(fopath);
            logf = open(os.path.join(fopath,'_000_000_log.txt'),'w');
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
                    imt = Image.fromarray((smsk_lst[-1]*255).astype(np.uint8));
                    imt.save(os.path.join(fopath,'_000_%03d_msk.png'%(len(smsk_lst)-1)));
            #
            img = np.stack(img_lst,axis=0);
            smsk = np.stack(smsk_lst,axis=0);
            tmsk = np.stack(tmsk_lst,axis=0);
            bdata.append(torch.from_numpy(img).cuda());
            bdata.append(torch.from_numpy(smsk).cuda());
            bdata.append(torch.from_numpy(tmsk).cuda());
            with torch.no_grad():
                boxout = boxnet(bdata);
            #
            size = np.prod(boxout['ss'].data.cpu().numpy(),axis=1);
            undone_queue = [];
            for idx,v in enumerate(size):
                heapq.heappush(undone_queue,(-v,idx));
            done_queue = [];
            msk_in = [];
            box_out = [];
            box_color = [];
            box_gt = [];
            s_gt = [];
            s_out = [];
            t_gt = [];
            t_out = [];
            r1_gt = [];
            r1_out = [];
            r2_gt = [];
            r2_out = [];
            baset = np.zeros([len(box_lst),3],dtype=np.float32);
            while len(undone_queue) > 0:
                if len(done_queue) > 0 :
                    itop = heapq.heappop(done_queue);
                    ci = itop[1];
                    bt = baset[ci,:][np.newaxis,:];
                    print('[done:%d'%ci,file=logf,end='');
                else:
                    itop = heapq.heappop(undone_queue);
                    ci = itop[1];
                    bo = boxout['sb'].data.cpu().numpy()[ci,...];
                    bgt,t = parsegt(box_lst[ci]);
                    add_gt_for_eval(box_lst[ci],s_gt,t_gt,r1_gt,r2_gt);
                    add_out_for_eval(
                        boxout['ss'].data[ci,...],
                        torch.from_numpy(t.astype(np.float32)).cuda(),
                        boxout['sr1'].data[ci,...],
                        boxout['sr2'].data[ci,...],
                        s_out,
                        t_out,
                        r1_out,
                        r2_out
                        );
                    baset[ci,:] = t[0,:];
                    bt = t;
                    box_out.append(bo+bt);
                    box_gt.append(bgt)
                    box_color.append(red_box);
                    msk_in.append(bdata[1].data.cpu().numpy()[ci,...]);
                    print('[undone:%d'%ci,file=logf,end='');
                unfinish = [];
                while len(undone_queue) > 0:
                    tdata = [];
                    tptdata = [];
                    jtop = heapq.heappop(undone_queue);
                    cj = jtop[1];
                    tdata.append(  bdata[0][0,...].unsqueeze(0) );
                    tdata.append( (bdata[1][ci,...]).unsqueeze(0) );
                    tdata.append( (bdata[1][cj,...]).unsqueeze(0) );
                    with torch.no_grad():
                        touchout = touchnet(tdata);
                    if touchout['y'].data.cpu().numpy()[0][0] > 0.5:
                        tptdata.append(bdata[0][0,...].unsqueeze(0));
                        tptdata.append((bdata[1][ci,...]).unsqueeze(0));
                        tptdata.append((bdata[1][cj,...]).unsqueeze(0));
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
                        w1 = tptout['w1'].data.cpu().numpy();
                        w2 = tptout['w2'].data.cpu().numpy();
                        box_out.append(bo+t+bt);
                        baset[cj,:] = (t+bt)[0,:];
                        box_gt.append(bgt);
                        box_color.append(blue_box);
                        msk_in.append(bdata[1].data.cpu().numpy()[cj,...]);
                        add_gt_for_eval(box_lst[cj],s_gt,t_gt,r1_gt,r2_gt);
                        add_out_for_eval(
                            boxout['ss'].data[cj,...],
                            torch.from_numpy((t+bt).astype(np.float32)).cuda(),
                            boxout['sr1'].data[cj,...],
                            boxout['sr2'].data[cj,...],
                            s_out,
                            t_out,
                            r1_out,
                            r2_out
                            );
                        print('->%d'%cj,file=logf,end='');
                    else:
                        unfinish.append(jtop);
                for p in unfinish:
                    heapq.heappush(undone_queue,p);
                print(']',file=logf);
            im = Image.fromarray((bdata[0][0,...].data.cpu().numpy()*255).astype(np.uint8));
            im.save(os.path.join(fopath,'_001_000_im.png'));
            writegt(fopath,box_gt);
            writeout(fopath,box_out,box_color,msk_in);
            val = eval(s_out,t_out,r1_out,r2_out,s_gt,t_gt,r1_gt,r2_gt);
            acc = {'cd':val.data.cpu().numpy()};
            json.dump(acc,os.path.join(fopath,'meta.json'));
            eval_all_sum += val.data.cpu().numpy();
            eval_all_cnt += 1.0;
            eval_cat_sum += val.data.cpu().numpy();
            eval_cat_cnt += 1.0;
        print(cat+',%f'%(eval_cat_sum/eval_cat_cnt),file=eval_log);
        eval_log.flush();
    print('inst mean,%f'%(eval_all_sum/eval_all_cnt),file=eval_log);
    eval_log.close();
            