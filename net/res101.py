import torch.nn as nn;
from .resnet import resnet101 as resnet

class Net(nn.Module):
    def __init__(self,**kargs):
        super(Net,self).__init__();
        self.net = resnet(num_classes=10);
    def forward(self,data):
        img = data[0];
        out = self.net(img);
        return out;