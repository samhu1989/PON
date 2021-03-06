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
        # z parameter
        self.fc_mu = nn.Linear(self.latent_size,self.z_size)
        self.fc_var = nn.Linear(self.latent_size,self.z_size)
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
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self,x,z):
        y = torch.cat([z,x],dim=1).contiguous();
        return self.decoder(y);

    def forward(self,input):
        x = input[0];
        mu, logvar = self.encode(x);
        z1 = self.sample(mu, logvar);
        z2 = self.sample(mu, logvar);
        xpart = x[:,self.part_idx].contiguous();
        rx1 = self.decode(xpart,z1);
        rx2 = self.decode(xpart,z2);
        out = {'rx1':rx1,'rx2':rx2,'mu':mu,'logvar':logvar};
        return out;