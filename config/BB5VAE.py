import os;
import sys;
import torch;
import pandas as pd;
from util.tools import write_tfb_loss;
from datetime import datetime;
import json;
import numpy as np;
from .config import NpEncoder;
from .BB1VAE import writelog,input_size,latent_size,z_size,workers,lr,weight_decay,nepoch,category;

beta = 5;

def loss(data,out):
    x = data[0];
    recon_x = out['rx']; 
    mu = out['mu'];
    logvar = out['logvar'];
    loss = {};
    loss['kl'] =  ( -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) ) / x.size(0);
    loss['recon'] = torch.sum( ( recon_x - x )**2 ) / x.size(0) ;
    loss['betakl'] = beta*loss['kl'];
    loss['overall'] = loss['betakl'] + loss['recon'];
    return loss;
    
    
def parameters(net):
    return net.parameters(); # train all parameters
    