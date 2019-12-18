import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__()
        self.input_size = kwargs['input_size'];
        self.latent_size = kwargs['latent_size'];
        self.z_size = kwargs['z_size'];
        self.beta = kwargs['beta'];
        self.part_idx = kwargs['part_idx'];
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,self.latent_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size//2,self.latent_size),
            nn.ReLU(inplace=True)
        );
        #
        self.fc = nn.Linear(self.latent_size,self.z_size)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.input_size-9+self.z_size,self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size,self.latent_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size//2,self.input_size),
        );

    def encode(self, x):
        x = self.encoder(x);
        x = self.fc(x);
        sx = F.sigmoid(x);
        if self.training:
            x = 2*sx*( 1 + x*(1-sx) ) - 1;
        else:
            x = torch.sign(x);
        return x;

    def decode(self,x,z):
        y = torch.cat([z,x],dim=1).contiguous();
        return self.decoder(y);

    def forward(self,input):
        x = input[0];
        z = self.encode(x);
        xpart = x[:,self.part_idx].contiguous();
        rx = self.decode(xpart,z);
        out = {'rx':rx};
        return out;