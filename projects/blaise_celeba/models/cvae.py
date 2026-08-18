"""CVAE (Conditional VAE) pour CelebA, conditionne par des attributs de visage.

Reference : Sohn, Lee & Yan, "Learning Structured Output Representation using
Deep Conditional Generative Models" (2015).

Meme squelette que le VAE (models/vae.py), avec en plus un vecteur de
condition `c` de dimension `num_conditions`. Chaque coordonnee de `c`
correspond a un attribut binaire CelebA (par exemple "Smiling"), et peut
valoir 0 ou 1 independamment des autres : c'est un vecteur multi-label
("multi-hot"), pas un one-hot exclusif comme la classe d'un chiffre MNIST.
Un visage peut tout a fait etre a la fois "Smiling" et "Male", alors qu'il ne
peut pas etre a la fois le chiffre 3 et le chiffre 8.

Choix d'injection de la condition : contrairement a l'implementation MNIST
de l'equipe (src/models/cvae.py), qui diffuse la condition comme des canaux
spatiaux constants concatenes a l'image, ce CVAE concatene le vecteur de
condition au vecteur de caracteristiques *apres* l'aplatissement de la
derniere carte de convolution, cote encodeur comme cote decodeur. Deux
raisons a ce choix :

1. Un attribut de visage (sourire, genre, couleur de cheveux) est une
   propriete globale de l'image entiere, pas un signal localise dans une
   region precise, contrairement a la forme d'un chiffre. Le concatener a un
   vecteur de caracteristiques deja global colle mieux a cette semantique
   qu'une carte spatiale.
2. Diffuser `num_conditions` canaux constants sur une image de 64x64 et les
   faire traverser 4 convolutions coute nettement plus cher en calcul, sur
   CPU, qu'ajouter quelques colonnes a une couche entierement connectee.
"""
from typing import Tuple

import torch
import torch.nn as nn

# Import relatif : voir la note dans models/vae.py (charge en script autonome
# et en import absolu depuis la demo web, seul un import relatif marche pour les deux).
from .layers import ConvBlock, ConvTransposeBlock, Flatten, Unflatten, compute_conv_output_size


class CVAE(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        image_size: Tuple[int, int] = (64, 64),
        latent_dim: int = 64,
        num_conditions: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.hidden_channels = hidden_channels

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

        # Condition concatenee apres l'aplatissement : la couche d'entree de
        # fc_mu/fc_logvar est donc plus large que encoder_output_dim.
        self.fc_mu = nn.Linear(self.encoder_output_dim + num_conditions, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim + num_conditions, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + num_conditions, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            Unflatten(c4, conv_out[0], conv_out[1]),
            ConvTransposeBlock(c4, c3),
            ConvTransposeBlock(c3, c2),
            ConvTransposeBlock(c2, c1),
            nn.ConvTranspose2d(c1, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        hidden = torch.cat([hidden, c], dim=1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c], dim=1)
        hidden = self.decoder_input(zc)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Genere `n` images pour un vecteur de condition donne.

        `c` est soit un vecteur 1D de longueur `num_conditions` (repete `n`
        fois), soit deja un batch (n, num_conditions).
        """
        device = next(self.parameters()).device
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(n, -1)
        c = c.to(device).float()
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, c)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Reconstruction deterministe : utilise mu directement, sans tirage aleatoire."""
        mu, _ = self.encode(x, c)
        return self.decode(mu, c)
