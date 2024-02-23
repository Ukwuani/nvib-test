import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim * 2)  # Output mean and log variance

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        latent_params = self.fc2(x)
        return latent_params
