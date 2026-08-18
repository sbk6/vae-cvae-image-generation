"""Blocs reseau reutilises par le VAE et le CVAE de ce sous-projet.

Ecrit independamment de src/models/layers.py (implementation MNIST). Le sujet
demande explicitement des implementations separees par dataset, et les
besoins different assez pour que reprendre ce code n'aurait rien simplifie :
visages couleur 64x64 ici, chiffres 28x28 en niveaux de gris la-bas.
"""
from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + LeakyReLU, bloc de base de l'encodeur.

    LeakyReLU(0.2) plutot qu'un ReLU classique : avec un reseau a 4 etages
    (contre 2 pour les chiffres MNIST), le risque d'unites mortes (neurones
    qui ne s'activent plus jamais) augmente. LeakyReLU laisse passer un
    gradient faible mais non nul cote negatif, ce qui limite ce risque sans
    changer le comportement qualitatif de l'activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTransposeBlock(nn.Module):
    """ConvTranspose2d + BatchNorm2d + ReLU, bloc de base des etages intermediaires du decodeur.

    Le tout dernier etage du decodeur n'utilise volontairement pas ce bloc
    (voir models/vae.py) : il applique directement ConvTranspose2d puis Tanh,
    sans BatchNorm ni ReLU intermediaire. Raison : dans l'implementation
    MNIST de l'equipe, le dernier bloc de decodage applique un ReLU avant le
    Tanh final, ce qui rend la moitie basse de la plage de sortie [-1, 1]
    inatteignable (documente dans backend/adapters/sylvain_mnist.py, section
    "Limite connue"). En separant le dernier etage, on evite de reproduire
    cette limite ici : la plage de sortie nominale du Tanh est reellement
    atteignable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.channels, self.height, self.width)


def compute_conv_output_size(
    input_size: Tuple[int, int],
    num_layers: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
) -> Tuple[int, int]:
    """Taille spatiale apres `num_layers` convolutions stride 2 identiques.

    Formule standard de convolution (independante du dataset), donc reprise
    telle quelle plutot que reinventee : (in + 2*padding - kernel) / stride + 1,
    appliquee `num_layers` fois.
    """
    height, width = input_size
    for _ in range(num_layers):
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1
    return height, width
