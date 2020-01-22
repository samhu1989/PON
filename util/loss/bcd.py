import torch;
import sys;
import numpy as np;
from net.cageutil import rot9,sr2box;
import pandas as pd;
from util.data.ply import write_ply, read_ply;
from util.data.gen_toybox import box_face;
from chamferdist import ChamferDistance;
distChamfer =  ChamferDistance();

bcd_pts = np.zeros([1200,3],dtype=np.float32);
span = np.random.uniform(-1.0,1.0,[200,2]);
index = [0,200,400,600,800,1000,1200];
span_dim = [[0,1],[0,1],[1,2],[1,2],[2,0],[2,0]];
fix_dim = [2,2,0,0,1,1];
fix_val = [1.0,-1.0,1.0,-1.0,1.0,-1.0];

for fi in range(len(index) - 1):
    start = index[fi];
    end = index[fi+1]; 
    bcd_pts[start:end,span_dim[fi]] = span;
    bcd_pts[start:end,fix_dim[fi]] = fix_val[fi];

def size2w(size):
    w = [];
    for fi in range(len(index) - 1):
        wtmp = torch.prod( size[:,span_dim[fi]].contiguous(), dim=1 );
        wtmp = wtmp.unsqueeze(1).contiguous();
        wtmp = wtmp.repeat(1,200);
        w.append(wtmp);
    w = torch.cat(w,dim=1);
    wm,_ = torch.max(w,dim=1,keepdim=True);
    return w / wm;
    
def sr2pts(size,r1,r2):
    pts = torch.from_numpy(bcd_pts);
    pts = pts.cuda();
    pts.requires_grad = True;
    rot = rot9(r1,r2);
    pts = pts*( size.unsqueeze(1).contiguous() );
    pts = torch.matmul(pts,rot);
    return pts;

def box_cd(osize,or1,or2,gtsize,gtr1,gtr2):
    opts = sr2pts(osize,or1,or2);
    gtpts = sr2pts(gtsize,gtr1,gtr2);
    dist1, dist2, idx1, idx2 = distChamfer(opts,gtpts);
    w = size2w(gtsize);
    w1 = torch.gather(w,1,idx1.long());
    ax1 = [x for x in range(1,dist1.dim())];
    ax2 = [x for x in range(1,dist2.dim())];
    cd_loss = torch.mean(dist1*w1,dim=ax1)+torch.mean(dist2*w,dim=ax2);
    return cd_loss;
    
def write_box(prefix,box):
    T=np.dtype([("n",np.uint8),("i0",np.int32),('i1',np.int32),('i2',np.int32)]);
    fidx = box_face;
    face = np.zeros(shape=[fidx.shape[0]],dtype=T);
    for i in range(fidx.shape[0]):
        face[i] = (3,fidx[i,0],fidx[i,1],fidx[i,2]);
    for i in range(box.shape[0]):
        write_ply(prefix+'_%d.ply'%i,points=pd.DataFrame(box[i,...]),faces=pd.DataFrame(face));

def run(**kwargs):
    nepoch = kwargs['nepoch'];
    bs = kwargs['batch_size'];
    size_gt = torch.from_numpy(np.random.uniform(0.1,1,[bs,3]).astype(np.float32));
    r1_gt = torch.from_numpy(np.random.uniform(-1,1,[bs,3]).astype(np.float32));
    r2_gt = torch.from_numpy(np.random.uniform(-1,1,[bs,3]).astype(np.float32));
    size = torch.zeros([bs,3],requires_grad=True);
    r1 = torch.zeros([bs,3],requires_grad=True);
    r2 = torch.zeros([bs,3],requires_grad=True);
    size.data.uniform_(0.1,1);
    r1.data.normal_(0.0,1.0);
    r2.data.normal_(0.0,1.0);
    opt = torch.optim.Adam([size,r1,r2],lr=0.001);
    boxgt = sr2box(size_gt.cuda(),r1_gt.cuda(),r2_gt.cuda());
    write_box('./log/boxcd/gt',boxgt.data.cpu().numpy());
    for i in range(nepoch):
        opt.zero_grad();
        cd = box_cd(size.cuda(),r1.cuda(),r2.cuda(),size_gt.cuda(),r1_gt.cuda(),r2_gt.cuda());
        loss = torch.mean(cd);
        loss.backward();
        print(loss.data.cpu().numpy());
        opt.step();
        boxout = sr2box(size.cuda(),r1.cuda(),r2.cuda());
        if i%100 == 0:
            write_box('./log/boxcd/out_%d'%i,boxout.data.cpu().numpy());
    return;