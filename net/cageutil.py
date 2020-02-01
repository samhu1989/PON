import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;

def rot6d(x_raw,y_raw):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    a1 = x_raw;
    a2 = y_raw;
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1);
    b3 = torch.cross(b1, b2, dim=1);
    return torch.stack((b1, b2, b3), dim=-1)

def rot9(x_raw,y_raw):
    x = F.normalize(x_raw,dim=1,p=2);
    z = torch.cross(x,y_raw,dim=1);
    z = F.normalize(z,dim=1,p=2);
    y = torch.cross(z,x,dim=1);
    rot = torch.stack([x,y,z],dim=1);
    rot = rot.view(-1,3,3);
    return rot;
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
    
def rot9np(x_raw,y_raw):
    x = normalize(x_raw);
    z = np.cross(x,y_raw);
    z = normalize(z);
    y = np.cross(z,x);
    rot = np.stack([x,y,z]);
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