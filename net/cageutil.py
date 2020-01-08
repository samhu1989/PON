import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;

def rot9(x_raw,y_raw):
    x = F.normalize(x_raw,dim=1,p=2);
    z = torch.cross(x,y_raw);
    z = F.normalize(z,dim=1,p=2);
    y = torch.cross(z,x);
    rot = torch.stack([x,y,z],dim=1);
    rot = rot.view(-1,3,3);
    return rot;
    
def add_msk_inst(obj,x,ms,mt):
    if obj.mode == 'full':
        x1 = torch.cat([x,ms],axis=1);
        x2 = torch.cat([x,mt],axis=1);
    elif obj.mode == 'part':
        x1 = torch.cat([x*ms,ms],axis=1);
        x2 = torch.cat([x*mt,mt],axis=1);
    else:
        assert False, "Unkown mode";
    return x1,x2;
    
def add_msk_dual(obj,x,ms,mt):
    if obj.mode == 'full':
        x1 = torch.cat([x,ms],axis=1);
        x2 = torch.cat([x,mt],axis=1);
    elif obj.mode == 'part':
        part = x*(ms+mt);
        x1 = torch.cat([part,ms],axis=1);
        x2 = torch.cat([part,mt],axis=1);
    else:
        assert False, "Unkown mode";
    return x1,x2;
    
def sr2box(size,r1,r2):
    const = np.array([[[1,1,-1],[-1,1,-1],[-1,1,1],[1,1,1],[1,-1,-1],[-1,-1,-1],[-1,-1,1],[1,-1,1]]],dtype=np.float32);
    const = torch.from_numpy(const);
    const = const.cuda();
    const.requires_grad = True;
    rot = rot9(r1,r2);
    box = const*( size.unsqueeze(1).contiguous() );
    box = torch.matmul(box,rot);
    return box;