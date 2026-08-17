"""Protocole commun aux differentes familles de modeles.

Le depot contient deux implementations independantes de VAE/CVAE, ecrites par
deux membres de l'equipe et volontairement laissees intactes :

- `src/`                          -> MNIST, sortie Tanh dans [-1, 1]
- `projects/david_fashion_mnist/` -> Fashion-MNIST, sortie Sigmoid dans [0, 1]

Elles different sur presque tout : normalisation des donnees, plage de sortie
du decodeur, signature de `forward()` (3 valeurs contre 4), maniere d'injecter
la condition (canaux spatiaux contre concatenation d'un vecteur one-hot).

Plutot que de reecrire l'une pour la faire ressembler a l'autre — ce qui
invaliderait les checkpoints deja entraines — chaque famille est enveloppee
dans un adaptateur exposant l'interface ci-dessous. Le reste du backend ne
connait que cette interface.

Le point le plus important est `output_range` : une image renvoyee dans
[0, 1] mais interpretee comme du [-1, 1] donne une image delavee **sans lever
la moindre erreur**. La plage est donc une propriete de l'adaptateur, jamais
une constante globale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


class ModelAdapter(ABC):
    """Enveloppe un modele entraine derriere une interface uniforme."""

    def __init__(self, module: torch.nn.Module, device: torch.device) -> None:
        self.module = module
        self.device = device
        self.module.to(device)
        self.module.eval()

    # ------------------------------------------------------------- metadata

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimension de l'espace latent."""

    @property
    @abstractmethod
    def is_conditional(self) -> bool:
        """Vrai si le modele attend une classe en plus du vecteur latent."""

    @property
    @abstractmethod
    def output_range(self) -> Tuple[float, float]:
        """Plage de valeurs produite par le decodeur, pour la conversion en PNG."""

    @property
    @abstractmethod
    def input_range(self) -> Tuple[float, float]:
        """Plage attendue en entree de l'encodeur (normalisation du dataset)."""

    @property
    def num_conditions(self) -> int:
        """Nombre de classes conditionnables (0 si le modele n'est pas conditionnel)."""
        return 0

    # ------------------------------------------------------------ inference

    @abstractmethod
    def encode(self, x: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        """Encode un batch d'images et renvoie mu (encodage deterministe)."""

    @abstractmethod
    def decode(self, z: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        """Decode un batch de vecteurs latents vers des images dans `output_range`."""

    # -------------------------------------------------------------- helpers

    def sample_latent(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        """Tire n vecteurs latents dans la prior N(0, I).

        Un `torch.Generator` dedie est utilise quand une seed est fournie, pour
        rendre le tirage reproductible sans perturber le RNG global du process.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        return torch.randn(n, self.latent_dim, device=self.device, generator=generator)

    def prepare_input(self, images_uint8: np.ndarray) -> torch.Tensor:
        """Convertit des images brutes (H, W) ou (N, H, W) en uint8 vers `input_range`.

        Le fixture stocke les images MNIST / Fashion-MNIST en uint8 [0, 255].
        Chaque famille de modeles attend ensuite sa propre normalisation.
        """
        array = np.asarray(images_uint8, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[None, None, :, :]
        elif array.ndim == 3:
            array = array[:, None, :, :]

        low, high = self.input_range
        # [0, 1] -> [low, high]
        array = array * (high - low) + low
        return torch.from_numpy(array).to(self.device)

    def to_display_range(self, images: torch.Tensor) -> torch.Tensor:
        """Ramene des images de `output_range` vers [0, 1] pour l'encodage PNG."""
        low, high = self.output_range
        return ((images - low) / (high - low)).clamp(0.0, 1.0)

    def validate_latent(self, values: Sequence) -> torch.Tensor:
        """Valide un vecteur latent recu en JSON.

        Flask ne fait aucune validation de schema (contrairement a Pydantic) :
        sans ce controle, un payload malforme remonterait en 500 depuis torch
        au lieu d'un 400 lisible.
        """
        if not isinstance(values, (list, tuple)):
            raise ValueError("'z' doit etre une liste de nombres.")
        if len(values) != self.latent_dim:
            raise ValueError(
                f"'z' doit contenir exactement {self.latent_dim} valeurs (recu {len(values)})."
            )

        cleaned: List[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"'z[{index}]' doit etre un nombre (recu {type(value).__name__}).")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"'z[{index}]' doit etre fini (NaN et inf interdits).")
            cleaned.append(value)

        return torch.tensor(cleaned, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def generate(self, n: int, class_label: Optional[int] = None, seed: Optional[int] = None) -> torch.Tensor:
        """Echantillonne la prior puis decode."""
        return self.decode(self.sample_latent(n, seed), class_label)

    @torch.no_grad()
    def interpolate(
        self,
        z_start: torch.Tensor,
        z_end: torch.Tensor,
        steps: int,
        class_label: Optional[int] = None,
    ) -> torch.Tensor:
        """Interpolation lineaire entre deux latents, decodee en une seule passe.

        Les `steps` points sont construits d'un coup et decodes en batch : une
        passe forward au lieu de `steps`, ce qui garde le slider fluide.
        """
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device).view(steps, 1)
        z = (1.0 - alphas) * z_start.view(1, -1) + alphas * z_end.view(1, -1)
        return self.decode(z, class_label)
