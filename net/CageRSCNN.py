import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from net.resnet import BasicBlock;
from torch.nn import functional as F; 
import numpy as np;
class BackBone(nn.Module):#
    def __init__(self,block,layers,input_channel=3):
        self.inplanes = 64
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,bias=False);
        self.bn1 = nn.BatchNorm2d(64);
        self.relu = nn.ReLU(inplace=True);
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1);
        self.layer1 = self._make_layer(block, 64, layers[0]);
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2);
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2);
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2);

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
                nn.BatchNorm2d((planes * block.expansion,affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers);

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x);#56x56x64
        #
        x1 = self.layer1(x); #56x56x64
        x2 = self.layer2(x1);#28x28x128
        x3 = self.layer3(x2);#14x14x256
        x4 = self.layer4(x3);#7x7x512
        #
        x2up = F.interpolate(x2,size=[56,56],mode='bilinear');
        x3up = F.interpolate(x3,size=[56,56],mode='bilinear');
        x4up = F.interpolate(x4,size=[56,56],mode='bilinear');
        feat = torch.cat([x1,x2up,x3up,x4up],dim=1);
        return feat;
        
class StopBone(nn.Module):#
    def __init__(self,block,layers,input_channel=3):
        self.inplanes = 64
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=1,bias=False);
        self.bn1 = nn.BatchNorm2d(64);
        self.relu = nn.ReLU(inplace=True);
        self.layer1 = self._make_layer(block, 64, layers[0]);
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2);
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2);
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256 * block.expansion, 2);
        self.act = nn.Softmax(dim=-1);

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
                nn.BatchNorm2d(planes * block.expansion,affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers);

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #
        x1 = self.layer1(x); #28x28x64
        x2 = self.layer2(x1);#14x14x128
        x3 = self.layer3(x2);#7x7x256
        y = self.avgpool(x3);#
        y = self.fc(y);
        #
        return self.act(y);
        
        
class SupressNet(nn.Module):
    def __init__(self):
        super(SupressNet, self).__init__();
        
        self.convs = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim,dim,kernel_size=1,bias=False)
            );
    
    def forward(self,feat,w,z):
        feat = feat*w.view(1,1,feat.size(2),feat.size(3));
        feat = self.convs(feat);
        feat,_ = torch.max(feat.view(feat.size(0),feat.size(1),-1),dim=-1);
        feat = torch.cat([feat,z],dim=1);
        return feat;
        
        
class StopNet(nn.Module):
    def __init__(self):
        super(StopNet,self).__init__();
        self.stopbone = StopBone(BasicBlock, [1, 1, 1, 1])
        
    def forward(self,dm):
        w = 1.0 - dm[:,1,:,:].contiguous();
        w = w.view(dm.size(0),1,dim.size(2),dim.size(3));
        stop = self.stopbone(dm);
        return w,stop;
        
        
        
class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet,self).__init__();
        self.f1 = nn.Sequential(
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, kernel_size=1),
            nn.Tanh()
        );
        
        self.f2 = nn.Sequential(
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, kernel_size=1),
            nn.Tanh()
        );
        
        self.inv1 = nn.Sequential(
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, kernel_size=1),
            nn.Tanh()
        );
        
        self.inv2 = nn.Sequential(
            nn.Conv1d(512+3, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, kernel_size=1),
            nn.Tanh()
        );
        
    def forward(self,f,grid):
        expf = f.view(-1,1,f.size(1)).repeat(1,x.size(1),1);
        fy1 = torch.cat([grid,expf],dim=1);
        y1 = self.f1(fy1);
        fy2 = torch.cat([y1,expf],dim=1);
        yout = self.f2(fy2);
        #
        finv1 = torch.cat([yout,expf],dim=1);
        yinv1 = self.inv1(finv1);
        finv2 = torch.cat([yinv1,expf],dim=1);
        yinv = self.inv2(finv2);
        #
        return yout,yinv;
       
class Net(nn.Module):#Net
    def __init__(self,**kwargs):
        super(Net,self).__init__();
        self.pts_num = kwargs['pts_num']
        self.backbone = BackBone(BasicBlock, [2, 2, 2, 2]);
        self.supress = SupressNet();
        self.deform = DeformNet();
        self.stop = StopNet();
        self._init_layers();
        
    def forward(self,*input):
        im = input[0];
        dm = input[1];
        if len(input) > 2: 
            z = input[2];
            grid = input[3];
        else:
            z = self.rand_z();
            grid = self.rand_grid();
        f = self.backbone(im);
        w,stop = self.stop(dm);
        sf = self.supress(f,w,z);
        y = self.deform(sf,grid);
        return y,stop;
        
    def rand_grid(self,x):
        rand_grid = torch.FloatTensor(x.size(0),3,self.pts_num);
        rand_grid.normal_(0.0,1.0);
        rand_grid += 1e-9;
        rand_grid = rand_grid / torch.norm(rand_grid,p=2.0,dim=1,keepdim=True);
        return rand_grid.type(x.type());
        
    def rand_z(self,x):
        rand_z = torch.FloatTensor(x.size(0),64);
        rand_z.normal_(0.0,1.0);
        return rand_z.type(x.type());
        
    def _init_layers(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels;
                m.weight.data.normal_(0,np.sqrt(2./n));
            elif isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0.0,0.02);
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.normal_(1.0,0.02);
                m.bias.data.fill_(0)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1);
                m.bias.data.zero_();
    
        
        
        
        