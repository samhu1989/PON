import os;
import sys;
import torch;
import pandas as pd;
from util.tools import write_tfb_loss;
from datetime import datetime;
import json;
import numpy as np;
from .config import NpEncoder;
from .BB1VAE import writelog,input_size,latent_size,z_size,workers,lr,weight_decay,nepoch,category,part_idx,loss,parameters;
beta = 5;
from functools import partial;
loss = partial(loss,beta=beta);
    