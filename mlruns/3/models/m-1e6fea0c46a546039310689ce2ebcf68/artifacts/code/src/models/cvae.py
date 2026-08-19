from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import ConvDecoderBlock, ConvEncoderBlock, Flatten, Unflatten, compute_conv_output_size


class CVAE(nn.Module):
    """Conditional VAE générique supportant `one_hot` et `multi_label`.

    - La condition est injectée en entrée de l'encodeur en tant que canaux supplémentaires: on
      transforme le vecteur de condition en cartes constantes et on concatène le long des canaux.
    - Le vecteur de condition est aussi concaténé au code latent z avant le décodeur.

    Avantages: cette méthode est agnostique à la taille d'image et fonctionne pour
    one-hot (une classe active) et multi-label (plusieurs attributs actifs).
    """

    def __init__(
        self,
        channels: int,
        image_size: Tuple[int, int],
        latent_dim: int,
        num_conditions: int,
        condition_mode: str = "one_hot",
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.condition_mode = condition_mode
        self.hidden_channels = hidden_channels

        in_channels = channels + num_conditions

        self.encoder = nn.Sequential(
            ConvEncoderBlock(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            ConvEncoderBlock(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            Flatten(),
        )

        conv_out_size = compute_conv_output_size(image_size, num_layers=2)
        self.encoder_output_dim = hidden_channels * 2 * conv_out_size[0] * conv_out_size[1]

        # proj condition to latent when concatenating
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # decoder input expects latent + condition
        self.decoder_input = nn.Linear(latent_dim + num_conditions, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            Unflatten(hidden_channels * 2, conv_out_size[0], conv_out_size[1]),
            ConvDecoderBlock(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvDecoderBlock(hidden_channels, channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Tanh(),
        )

    def _prepare_condition_maps(self, c: Union[torch.Tensor, int], batch_size: int, device: torch.device) -> torch.Tensor:
        """Transforme la condition en cartes de taille (batch, num_conditions, H, W).

        Accepte :
        - un entier (classe) si `condition_mode` == 'one_hot'
        - un tenseur de forme (batch,) d'entiers
        - un tenseur de forme (batch, num_conditions) de 0/1 pour multi-label
        """
        H, W = self.image_size
        if isinstance(c, int):
            vec = torch.zeros(batch_size, self.num_conditions, device=device)
            vec[:, c] = 1.0
        elif isinstance(c, torch.Tensor):
            if c.dim() == 1:
                # indices
                vec = F.one_hot(c.long(), num_classes=self.num_conditions).float()
            else:
                # assume already multi-hot or one-hot vectors
                vec = c.float()
        else:
            raise ValueError("Condition type not supported")

        maps = vec.view(batch_size, self.num_conditions, 1, 1).expand(batch_size, self.num_conditions, H, W)
        return maps, vec

    def encode(self, x: torch.Tensor, c: Union[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        batch_size = x.size(0)
        cond_maps, _ = self._prepare_condition_maps(c, batch_size, device)
        x_cond = torch.cat([x, cond_maps], dim=1)
        hidden = self.encoder(x_cond)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c_vec: torch.Tensor) -> torch.Tensor:
        # c_vec should be shape (batch, num_conditions)
        zc = torch.cat([z, c_vec], dim=1)
        hidden = self.decoder_input(zc)
        x_hat = self.decoder(hidden)
        return x_hat

    def forward(self, x: torch.Tensor, c: Union[torch.Tensor, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        # prepare condition vector (batch, num_conditions)
        batch_size = x.size(0)
        _, c_vec = self._prepare_condition_maps(c, batch_size, x.device)
        x_hat = self.decode(z, c_vec)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, c: Union[torch.Tensor, int], n: int = 1) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = n
        # prepare condition vector
        _, c_vec = self._prepare_condition_maps(c, batch_size, device)
        z = torch.randn(batch_size, self.latent_dim, device=device)
        x_hat = self.decode(z, c_vec)
        return x_hat

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, c: Union[torch.Tensor, int]) -> torch.Tensor:
        mu, logvar = self.encode(x, c)
        z = mu  # use mean for deterministic reconstruction
        _, c_vec = self._prepare_condition_maps(c, x.size(0), x.device)
        x_hat = self.decode(z, c_vec)
        return x_hat
