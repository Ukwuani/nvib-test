import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder

class VIB(nn.Module):
    def __init__(self,
                input_dim, 
                latent_dim,
                prior_mu = None,
                prior_var = None,
                prior_alpha = None,
                delta=1,
                kappa=1,
                beta=1.0
                 ):
        super(VIB, self).__init__()
        # init priors
        self.prior_mu = prior_mu if prior_mu is not None else torch.zeros(input_dim) #Prior for Gaussian means μ^p
        self.prior_var = prior_var if prior_var is not None else torch.ones(input_dim) #Prior for Gaussian variance (σ^2)^p
        self.prior_alpha = prior_alpha if prior_alpha is not None else torch.zeros(1) #prior for Dirichlet Psuedo-counts α^p 
        self.delta = float(delta)  # Conditional prior delta  α^δ
        self.kappa = int(kappa)  # Number of samples   k^δ

        # init layers
        self.input_dim = input_dim # input dimension
        self.relu = nn.ReLU()  # Relu activation for alpha
        self.proj_mu = nn.Linear(input_dim, latent_dim)  # Project to mean
        self.proj_log_var = nn.Linear(input_dim, latent_dim)  # Project to log variance
        self.proj_alpha = nn.Linear(input_dim, 1)  # Project to model size
        
        self.encoder = Encoder(input_dim, latent_dim) # Encoder
        self.decoder = Decoder(latent_dim, input_dim) # Decoder
        self.beta = beta

    def forward(self, x):
        # Encode the input data
        latent_params = self.encoder(x)

        # Sample from the latent distribution
        mu, log_var = torch.chunk(latent_params, 2, dim=-1)
        
         # Project to mean, log variance and log alpha
        mu = self.proj_mu(latent_params)
        log_var = self.proj_log_var(latent_params)
        alpha = self.relu(self.alpha_proj(latent_params))

        # Catering to Unknowns u
        u_mu = torch.ones_like(mu)[0, :, :].unsqueeze(0) * self.prior_mu
        u_log_var = torch.ones_like(log_var)[0, :, :].unsqueeze(
            0
        ) * math.log(self.prior_var)
        u_alpha = (
            torch.ones_like(alpha)[0, :, :].unsqueeze(0) * self.prior_alpha
        )
        mu = torch.cat((u_mu, mu))
        log_var = torch.cat((u_log_var, log_var))
        alpha = torch.cat((u_alpha, alpha))
        
        #  Reparameterization (guassian)
        z = mu
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z = mu + epsilon * std
            
        
        
        # Decode the latent representation
        recon_x = self.decoder(z)

        return recon_x, mu, log_var
