import math
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
import numpy as np


class VAE(nn.Module):
    def __init__(self, in_res=64, in_cnl=3, latent_cnl=4):
        super(VAE, self).__init__()

        self.in_res = in_res
        self.latent_cnl = latent_cnl

        # in_cnl x in_res x in_res -> 1024 x in_res//8 x in_res//8 
        self.CBR = nn.Sequential(
            nn.Conv2d(in_cnl, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128 ,kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        # 1024 x in_res//8 x in_res//8  -> latent_cnl*2 x in_res//8 x in_res//8
        self.flatten = nn.Sequential(
            nn.Conv2d(1024, latent_cnl*2, kernel_size=3, padding=1),
            nn.Conv2d(latent_cnl*2, latent_cnl*2, kernel_size=1)
        )
        
        # latent_cnl x in_res//8 x in_res//8 -> 1024 x in_res//8 x in_res//8
        self.expand = nn.Sequential(
            nn.Conv2d(latent_cnl, latent_cnl, kernel_size=1),
            nn.Conv2d(latent_cnl, 1024, kernel_size=3, padding=1)
        )
        
        # 1024 x in_res//8 x in_res//8 -> in_cnl x in_res x in_res 
        self.CTBR = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, in_cnl, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def encode(self, x):
        x = self.CBR(x)
        x = self.flatten(x)
        mu, logvar = torch.chunk(x, 2, dim=1)

        # flatten
        mu = mu.view(-1, self.latent_cnl * ((self.in_res//8)**2))
        logvar = logvar.view(-1, self.latent_cnl * ((self.in_res//8)**2))
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std

        # unflatten
        z = z.view(-1, self.latent_cnl, self.in_res//8, self.in_res//8)
        return z

    def decode(self, x):
        x = self.expand(x)
        x = self.CTBR(x)
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
