import math
import numpy as np;
from torch import nn
from torch.autograd import Function
import torch
import os
import sys
from numbers import Number
from collections import Set, Mapping, deque
import chamfer
import copy


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, dim = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize,n)
        dist2 = torch.zeros(batchsize,m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2

class chamferDist(nn.Module):
    def __init__(self):
        super(chamferDist, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)
        
def knn(xyz,k,debug=False,rdist=False):
    batchsize, n, _ = xyz.size()
    dist = torch.zeros(batchsize,n,k);
    dist = dist.cuda();
    idx = torch.zeros(batchsize,n,k,2).type(torch.IntTensor);
    idx = idx.cuda();
    k = torch.IntTensor(np.array([k]));
    chamfer.knn(xyz,k,dist,idx);
    if debug:
        print(dist.cpu().numpy());
        print(idx.cpu().numpy());
    if rdist:
        return dist,idx;
    else:
        return idx;

class interpFunction(Function):
    @staticmethod
    def forward(ctx,z,prob):
        b,p,dim,L = z.size();
        b1,p1,M,N = prob.size();
        assert dim == 2,'this implementation only carries out interpolation in 2D';
        assert b == b1,'the input have different batch size %d and %d'%(b,b1);
        assert p == p1,'the input have different patch number %d and %d'%(p,p1);
        #
        idx = torch.zeros(b,p,2,4,L).type(torch.IntTensor);
        idx = idx.cuda();
        w = torch.zeros(b,p,4,L).type(torch.FloatTensor);
        w = w.cuda();
        p = torch.zeros(b,p,L).type(torch.FloatTensor);
        p = p.cuda();
        chamfer.interp_forward(z,prob,idx,w,p);
        ctx.save_for_backward(z,prob,idx,w);
        return p;

    @staticmethod
    def backward(ctx,grad):
        z,prob,idx,w = ctx.saved_tensors;
        b,p,dim,L = z.size();
        b1,p1,M,N = prob.size();
        assert dim == 2,'this implementation only carries out interpolation in 2D';
        assert b == b1,'the input have different batch size %d and %d'%(b,b1);
        assert p == p1,'the input have different patch number %d and %d'%(p,p1);
        gradz = torch.zeros_like(z);
        gradp = torch.zeros_like(prob);
        chamfer.interp_backward(grad,idx,w,gradp);
        return gradz,gradp ;
        
selectN = 100;
class selectFunction(Function):
    @staticmethod
    def forward(ctx,input,selectbool):
        global selectN;
        size = z.size();
        outsize = copy.deepcopy(size);
        outsize[-1] = selectN;
        output = torch.zeros(*outsize).type(torch.FloatTensor);
        output = output.cuda();
        outidx = torch.zeros(size[0],size[1],selectN).type(torch.IntTensor);
        outidx = outidx.cuda();
        chamfer.select_forward(input,selectbool,outidx,output);
        ctx.save_for_backward(input,outidx);
        return output;

    @staticmethod
    def backward(ctx,outputgrad):
        input,outidx = ctx.saved_tensors;
        inputgrad = torch.zeros_like(input);
        chamfer.select_backward(outputgrad,outidx,inputgrad);
        return inputgrad;
