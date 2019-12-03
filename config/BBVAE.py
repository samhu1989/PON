import os;
import sys;
import torch;
import pandas as pd;
beta = 1;
input_size = 18;
z_size = 16;

def loss(data,out):
    x = data[0];
    recon_x = out['rx'];
    mu = out['mu'];
    logvar = out['logvar'];
    loss = {};
    loss['kl'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp());
    loss['recon'] = F.binary_cross_entropy(recon_x, x, reduction='sum');
    loss['overall'] = ( loss['kl'] + loss['recon'] ) / x.size(0);
    return loss;
    
    
def parameters(net):
    return net.parameters(); # train all parameters