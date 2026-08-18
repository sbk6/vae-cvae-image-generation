"""Adaptateur pour les modeles CelebA de `projects/blaise_celeba/`.

Specificites de cette famille (voir projects/blaise_celeba/README.md pour le
detail et les justifications) :
- images RGB 64x64, normalisees dans [-1, 1] ;
- decodeur termine par Tanh, sortie dans [-1, 1], reellement atteignable
  (contrairement au modele MNIST, voir models/layers.py du sous-projet) ;
- CVAE conditionne par un vecteur multi-hot de 3 attributs binaires
  (Smiling, Male, Wavy_Hair par defaut), concatene apres l'aplatissement des
  cartes de convolution, pas par des canaux spatiaux comme le modele MNIST ;
- `forward()` renvoie 3 valeurs ;
- pas de YAML externe requis : la configuration complete (dataset, modele,
  entrainement) est sauvegardee dans le checkpoint lui-meme sous la cle
  "configuration", donc chaque checkpoint est autonome, comme pour
  Fashion-MNIST.

Traduction class_label -> attributs : l'interface commune
(`backend/adapters/base.py`) attend un entier `class_label` unique, pensee a
l'origine pour une classe exclusive (le chiffre 0-9 de MNIST). Les attributs
CelebA ne sont pas exclusifs (un visage peut etre a la fois Male et Smiling),
donc `class_label` est ici traduit en une des `2 ** num_conditions`
combinaisons possibles du vecteur multi-hot (0..7 pour 3 attributs), plutot
que de modifier l'interface commune au risque de casser MNIST/Fashion-MNIST.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from backend.adapters.base import ModelAdapter
from projects.blaise_celeba.models.cvae import CVAE
from projects.blaise_celeba.models.vae import VAE


class BlaiseCelebaAdapter(ModelAdapter):
    """Enveloppe un VAE ou CVAE de `projects/blaise_celeba/models/`."""

    def __init__(self, module: torch.nn.Module, device: torch.device, attribute_names: List[str]) -> None:
        super().__init__(module, device)
        self.attribute_names = attribute_names
        self._combinations: List[Tuple[float, ...]] = list(
            itertools.product([0.0, 1.0], repeat=len(attribute_names))
        )

    @property
    def latent_dim(self) -> int:
        return int(self.module.latent_dim)

    @property
    def is_conditional(self) -> bool:
        return isinstance(self.module, CVAE)

    @property
    def num_conditions(self) -> int:
        return len(self._combinations) if self.is_conditional else 0

    @property
    def output_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    @property
    def input_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def combination_label(self, class_label: int) -> str:
        """Description humaine d'une combinaison (utilisee pour les class_names du catalogue)."""
        combo = self._combinations[class_label]
        active = [name for name, value in zip(self.attribute_names, combo) if value > 0]
        return ", ".join(active) if active else "aucun attribut marque"

    def _condition_vector(self, class_label: int, n: int) -> torch.Tensor:
        combo = self._combinations[class_label]
        vector = torch.tensor(combo, dtype=torch.float32, device=self.device)
        return vector.unsqueeze(0).expand(n, -1)

    def prepare_input(self, images_uint8: np.ndarray) -> torch.Tensor:
        """Surcharge : les fixtures CelebA sont en couleur, contrairement a MNIST/Fashion-MNIST.

        `ModelAdapter.prepare_input` (base.py) suppose une image a un seul
        canal, forme (H, W) ou (N, H, W) : appliquee telle quelle a du RGB
        (H, W, 3), elle interpreterait a tort le canal couleur comme un batch.
        """
        array = np.asarray(images_uint8, dtype=np.float32) / 255.0
        if array.ndim == 3:
            # (H, W, 3) -> (1, 3, H, W)
            array = array[None, :, :, :].transpose(0, 3, 1, 2)
        elif array.ndim == 4:
            # (N, H, W, 3) -> (N, 3, H, W)
            array = array.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Forme d'image inattendue pour CelebA (RGB attendu) : {array.shape}")

        low, high = self.input_range
        array = array * (high - low) + low
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            condition = self._condition_vector(class_label, x.size(0))
            mu, _ = self.module.encode(x, condition)
        else:
            mu, _ = self.module.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            condition = self._condition_vector(class_label, z.size(0))
            return self.module.decode(z, condition)
        return self.module.decode(z)


def load(
    checkpoint_path: Path,
    device: torch.device,
    **_ignored,
) -> BlaiseCelebaAdapter:
    """Charge un checkpoint CelebA. Autonome : la configuration vient du checkpoint, pas d'un YAML."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "configuration" not in checkpoint:
        raise ValueError(
            f"Checkpoint invalide : 'model_state_dict' et 'configuration' sont requis ({checkpoint_path})."
        )

    config = checkpoint["configuration"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    attribute_names = list(dataset_cfg["attributes"])

    common_kwargs = dict(
        channels=dataset_cfg["channels"],
        image_size=tuple(dataset_cfg["image_size"]),
        latent_dim=model_cfg["latent_dim"],
        hidden_channels=model_cfg["hidden_channels"],
    )

    if model_cfg.get("type", "vae") == "cvae":
        module = CVAE(num_conditions=len(attribute_names), **common_kwargs)
    else:
        module = VAE(**common_kwargs)

    module.load_state_dict(checkpoint["model_state_dict"])
    return BlaiseCelebaAdapter(module, device, attribute_names)


def describe(checkpoint_path: Path) -> Dict[str, Any]:
    """Lit les metadonnees d'un checkpoint sans reconstruire le modele complet."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("configuration") or {}
    return {
        "model_type": config.get("model", {}).get("type"),
        "beta": config.get("training", {}).get("beta"),
        "epoch": checkpoint.get("epoch"),
        "best_loss": checkpoint.get("best_loss"),
        "attributes": config.get("dataset", {}).get("attributes"),
    }
