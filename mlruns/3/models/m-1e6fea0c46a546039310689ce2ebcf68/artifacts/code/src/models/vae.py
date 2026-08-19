from typing import Tuple

import torch
import torch.nn as nn

from src.models.layers import ConvDecoderBlock, ConvEncoderBlock, Flatten, Unflatten, compute_conv_output_size


class VAE(nn.Module):
    def __init__(
        self,
        channels: int,
        image_size: Tuple[int, int],
        latent_dim: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ConvEncoderBlock(channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            ConvEncoderBlock(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            Flatten(),
        )

        conv_out_size = compute_conv_output_size(image_size, num_layers=2)
        self.encoder_output_dim = hidden_channels * 2 * conv_out_size[0] * conv_out_size[1]

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            Unflatten(hidden_channels * 2, conv_out_size[0], conv_out_size[1]),
            ConvDecoderBlock(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvDecoderBlock(hidden_channels, channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z)
        reconstructed = self.decoder(hidden)
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
