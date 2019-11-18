from __future__ import print_function
import os;
import h5py;
import numpy as np;
import torch
import torchvision
import torchvision.transforms as transforms
import torch;
import torch.utils.data as data;

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
class Data(torchvision.datasets.CIFAR10):
    def __init__(self,opt,train=True):
        super(Data,self).__init__(root=opt['data_path'],train=train,download=True,transform=transform);