import os;
import sys;
import torch;
import pandas as pd;

def loss(data,out):
    loss = {};
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp());
    return;
    
    
def loss(self, recon_x, x, mu, logvar):
    # reconstruction losses are summed over all elements and batch
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size