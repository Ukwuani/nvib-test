import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder

class VIB(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VIB, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        # Encode the input data q(z | x)
        latent_params = self.encoder(x)
        

        # Sample from the latent distribution
        mu, log_var = torch.chunk(latent_params, 2, dim=-1)
        
        # Gaussian Reparameterization
        z = self.reparameterize(mu, log_var)

        # Decode the latent representation q(x | z)
        recon_x = self.decoder(z)

        return recon_x, mu, log_var
