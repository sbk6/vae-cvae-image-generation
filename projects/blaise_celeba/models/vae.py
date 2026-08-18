"""VAE (Variational Autoencoder) pour des visages CelebA en couleur.

Reference : Kingma & Welling, "Auto-Encoding Variational Bayes" (2013),
https://arxiv.org/abs/1312.6114.

Une image passe dans l'encodeur, qui produit deux vecteurs `mu` et `logvar` :
la moyenne et le log de la variance d'une distribution gaussienne diagonale
q(z|x). On tire un point z dans cette distribution avec l'astuce de
reparametrisation (z = mu + eps * exp(0.5*logvar), eps ~ N(0, I)), ce qui
permet de retropropager le gradient malgre le tirage aleatoire. Le decodeur
reconstruit ensuite une image a partir de z.

Difference principale avec l'implementation MNIST (src/models/vae.py) : 4
etages de convolution au lieu de 2, et 3 canaux d'entree (RGB) au lieu d'1.
Des visages a 64x64 en couleur portent beaucoup plus d'information par image
qu'un chiffre a 28x28 en niveaux de gris (texture de peau, couleur des
cheveux, arriere-plan) ; un encodeur a 2 etages laisserait une carte de
caracteristiques encore trop grande (16x16) pour un vecteur latent compact,
et n'aurait pas assez de profondeur pour apprendre des motifs visuels aussi
varies.
"""
from typing import Tuple

import torch
import torch.nn as nn

# Import relatif : ce module est charge de deux facons differentes (en script
# autonome via `python -m ...` depuis projects/blaise_celeba/, et en import
# absolu `projects.blaise_celeba.models.vae` depuis la demo web), et seul un
# import relatif fonctionne dans les deux cas.
from .layers import ConvBlock, ConvTransposeBlock, Flatten, Unflatten, compute_conv_output_size


class VAE(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        image_size: Tuple[int, int] = (64, 64),
        latent_dim: int = 64,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # 4 etages, doublant les canaux a chaque fois : 32 -> 64 -> 128 -> 256.
        # A 64x64 en entree, la carte de caracteristiques finale fait 4x4.
        c1, c2, c3, c4 = hidden_channels, hidden_channels * 2, hidden_channels * 4, hidden_channels * 8

        self.encoder = nn.Sequential(
            ConvBlock(channels, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
            ConvBlock(c3, c4),
            Flatten(),
        )

        conv_out = compute_conv_output_size(image_size, num_layers=4)
        self.encoder_output_dim = c4 * conv_out[0] * conv_out[1]

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            Unflatten(c4, conv_out[0], conv_out[1]),
            ConvTransposeBlock(c4, c3),
            ConvTransposeBlock(c3, c2),
            ConvTransposeBlock(c2, c1),
            # Dernier etage : pas de BatchNorm/ReLU avant le Tanh (voir
            # docstring de ConvTransposeBlock dans layers.py).
            nn.ConvTranspose2d(c1, channels, kernel_size=4, stride=2, padding=1),
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
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n: int = 1) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)
