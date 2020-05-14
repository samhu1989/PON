import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from net.resnet import BasicBlock;
import numpy as np;
from .rcnn.rpn.rpn import _RPN;

class BackBone(nn.Module):#
    def __init__(self,block,layers,input_channel=3):
        self.inplanes = 64
        self.norm = nn.BatchNorm2d;
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion,affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x);#56x56x64
        #
        x1 = self.layer1(x0); #56x56x64
        x2 = self.layer2(x1);#28x28x128
        x3 = self.layer3(x2);#14x14x256
        x4 = self.layer4(x3);#14x14x256
        return [x0,x1,x2,x3,x4];
        
    
class ZNet(nn.Module):#Z Axis Prediction
    def __init__(self):
        super(ZNet,self).__init__();
        self.layer1 = self._make_layer(768,128,stride=2);
        self.layer2 = self._make_layer(256,64,stride=2);
        self.layer3 = self._make_layer(128,64);
        self.layer3 = self._make_layer(128,64,stride=2);
        self.dconv = nn.ConvTranspose2d(64,32,3,stride=2);
        self.relu = nn.ReLU(inplace=True);
        self.conv = nn.Conv2d(32,2,kernel_size=3,padding=1);
    
    def _make_layer(self,inc,outc,stride=1):
        layers = [];
        layers.append(nn.Conv2d(inc, outc, kernel_size=1,bias=False));
        layers.append(nn.BatchNorm2d(outc));
        layers.append(nn.ReLU(inplace=True));
        if stride > 1:
            layers.append(nn.ConvTranspose2d(outc, outc, stride+1, stride=stride));
            layers.append(nn.ReLU(inplace=True));
        return nn.Sequential(*layers);
    
    def forward(self,xs):
        y1 = torch.cat([xs[3],xs[4]],dim=1);
        y1 = self.layer1(y1); #28x28x128
        y2 = torch.cat([y1,xs[2]],dim=1);
        y2 = self.layer2(y2); #56x56x64
        y3 = torch.cat([y2,xs[1]],dim=1);
        y3 = self.layer3(y3); #56x56x64
        y4 = torch.cat([y3,xs[0]],dim=1);
        y4 = self.layer4(y4); #112x112x64
        #
        y5 = self.dconv(y4); #224x224x32
        y5 = self.relu(y5);
        y5 = self.conv(y5); #224x224x2
        y5 = torch.exp(y5);
        return y5;
        
class FoldNet(nn.Module):
    def __init__(self):
        super(FoldNet,self).__init__();
        self.f1 = nn.Sequential(
            [
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, kernel_size=1),
            nn.Tanh()
            ]
        );
        
        self.f2 = nn.Sequential(
            [
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 6, kernel_size=1),
            nn.Softmax(dim=1)
            ]
        );
        
        self.inv1 = nn.Sequential(
            [
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, kernel_size=1),
            nn.Tanh()
            ]
        );
        
        self.inv2 = nn.Sequential(
            [
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, kernel_size=1),
            nn.Tanh()
            ]
        );
        
    def forward(self,x,f,cage):
        expf = f.view(-1,1,f.size(1)).repeat(1,x.size(1),1);
        
        y = torch.cat([x,expf],dim=1);
        self.f1(x);
        y = torch.cat([y,expf],dim=1);
        yw = self.f2(y);
        y = torch.sum(yw*cage,dim=1);#dim is not correct here
        #
        yinv = torch.cat([y,expf],dim=1);
        self.inv1(yinv);
        yinv = torch.cat([yinv,expf],dim=1);
        self.inv2(yinv);
        #
        return yw,y,yinv;
    

class CageNet(nn.Module):#CageNet
    def __init__(self):
        self.backbone = BackBone(BasicBlock, [2, 2, 2, 2]);
        self.rpn = _RPN();
        self.zaxis = ZNet();
        self.foldnet = FoldNet();
        
    def forward(self,*input):
        x = input[0];
        if x.dim() == 4:
            x = x[:,:3,:,:].contiguous();
        
        return 

        
        
        