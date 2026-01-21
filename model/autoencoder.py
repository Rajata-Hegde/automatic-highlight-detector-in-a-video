import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection

class AutoencoderRASL(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(768),

            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(384),

            nn.Linear(384, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(384),

            nn.Linear(384, 768),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(768),

            nn.Linear(768, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
