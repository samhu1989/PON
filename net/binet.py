import torch;
import torch.nn as nn;
import numpy as np;
import torch.nn.functional as F
import math;
import random;
#
class BiBlock(nn.Module):
    expansion = 4;
    def __init__(self, inplanes, planes, stride=1):
        super(BiBlock, self).__init__()
        num = int(math.sqrt(self.expansion*planes))
        #one bilinear branch
        self.conv1 = nn.Conv2d(inplanes//2,num,kernel_size=1,bias=False);
        self.bn1 = nn.BatchNorm2d(num);
        self.relu = nn.ReLU(inplace=True);
        self.conv2 = nn.Conv2d(num,num-1,kernel_size=3,stride=stride,padding=1);
        self.bn2 = nn.BatchNorm2d(num-1);
        #other bilinear branch
        self.conv3 = nn.Conv2d(inplanes//2,num-1,kernel_size=3,stride=stride,padding=1);
        self.bn3 = nn.BatchNorm2d(num-1);
        #identity branch
        if stride > 1 or inplanes != self.expansion*planes:
            self.conv4 = nn.Conv2d(inplanes,self.expansion*planes,kernel_size=1,stride=stride,bias=False);
            self.bn4 = nn.BatchNorm2d(self.expansion*planes);
        self.stride = stride;
        self.planes = planes;
        self.inplanes = inplanes;

    def forward(self, x):
        #
        x1 = x[:,[i for i in range(0,x.size(1),2)],:,:].contiguous();
        x1 = self.conv1(x1);
        x1 = self.bn1(x1);
        x1 = self.relu(x1);
        x1 = self.conv2(x1);
        x1 = self.bn2(x1);
        x1 = torch.cat([x1,torch.ones(x1.size(0),1,x1.size(2),x1.size(3)).type(x1.type())],dim=1);
        #
        y1 = x1.view(x1.size(0),x1.size(1),-1);
        y1 = y1.transpose(1,2).contiguous().view(-1,x1.size(1),1);
        #
        x2 = x[:,[i for i in range(1,x.size(1),2)],:,:];
        x2 = self.conv3(x2);
        x2 = self.bn3(x2);
        x2 = torch.cat([x2,torch.ones(x2.size(0),1,x2.size(2),x2.size(3)).type(x2.type())],dim=1);
        #
        y2 = x2.view(x2.size(0),x2.size(1),-1);
        y2 = y2.transpose(1,2).contiguous().view(-1,1,x2.size(1));
        #
        x12 = torch.bmm(y1,y2);
        x12 = x12.view(x1.size(0),x1.size(2),x1.size(3),-1);
        x12 = x12.permute(0,3,1,2).contiguous();
        #
        if self.stride > 1 or self.inplanes != self.expansion*self.planes:
            x3 = self.conv4(x);
            x3 = self.bn4(x3);
        else:
            x3 = x;
        #
        out = x12 + x3;
        out = self.relu(out)
        return out;
#
class BiNet(nn.Module):
    def __init__(self, block, layers,input_channel=3 ,num_classes=1000, fc=True):
        self.inplanes = 64
        super(BiNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 121, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 484, layers[3], stride=2)
        self.dofc = fc;
        if fc:
            self.fc = nn.Linear(484*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride));
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.dofc:
            if x.size(2) > 1 or x.size(3) > 1:
                x = torch.mean(x,dim=[2,3]);
            x = x.view(x.size(0),-1);
            x = self.fc(x);
        return x;
        
def binet101(**kwargs):
    model = BiNet(BiBlock,[3, 4, 23, 3],**kwargs)
    return model
#
class Net(nn.Module):
    def __init__(self,**kargs):
        super(Net,self).__init__();
        self.net = binet101(num_classes=10);
    def forward(self,data):
        img = data[0];
        out = self.net(img);
        return out;
