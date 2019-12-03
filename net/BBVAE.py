import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Net(nn.Module):
    def __init__(self,**kwargs):
        super(Net, self).__init__()
        self.input_size = kwargs['input_size'];
        self.latent_size = kwargs['latent_size'];
        self.z_size = kwargs['z_size'];
        self.beta = kwargs['beta'];
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size,self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size,self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True)
        );
        # z parameter
        self.fc_mu = nn.Linear(self.latent_size,self.z_size)
        self.fc_var = nn.Linear(self.latent_size,self.z_size)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.input_size-3+self.z_size,self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size,self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_size,self.input_size),
        );

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self,x,z):
        y = torch.cat([x,z],dim=1).contiguous();
        return self.decoder(y);

    def forward(self, x):
        mu, logvar = self.encode(x);
        z = self.sample(mu, logvar);
        xpart = x[:,3:].contiguous();
        rx = self.decode(xpart,z);
        out = {'rx':rx,'mu':mu,'logvar':logvar};
        return out;