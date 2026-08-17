"""Adaptateur pour les modeles MNIST de `src/`.

Specificites de cette famille :
- donnees normalisees dans [-1, 1] par transforms.Normalize((0.5,), (0.5,)) ;
- decodeur termine par Tanh, donc sortie nominale dans [-1, 1] ;
- CVAE conditionne par des cartes de condition concatenees aux canaux de
  l'image, et par un vecteur one-hot concatene a z avant le decodeur ;
- `forward()` renvoie 3 valeurs, et la configuration vit dans un YAML externe.

Limite connue heritee du modele : le dernier ConvDecoderBlock applique un ReLU
avant le Tanh, donc la sortie reelle est bornee dans [0, 1) et le fond noir
(-1) est inatteignable. On garde volontairement `output_range = (-1, 1)` : la
plage nominale de l'architecture est bien celle-la, et la retoucher ici
masquerait le bug au lieu de le montrer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from backend.adapters.base import ModelAdapter
from src.data.datasets import DatasetInfo
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


class SylvainMnistAdapter(ModelAdapter):
    """Enveloppe un VAE ou CVAE de `src/models/`."""

    @property
    def latent_dim(self) -> int:
        return int(self.module.latent_dim)

    @property
    def is_conditional(self) -> bool:
        return getattr(self.module, "num_conditions", None) is not None

    @property
    def num_conditions(self) -> int:
        return int(getattr(self.module, "num_conditions", 0) or 0)

    @property
    def output_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    @property
    def input_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)

    def _condition_vector(self, class_label: int, n: int) -> torch.Tensor:
        """Construit le vecteur one-hot (n, num_conditions) attendu par le decodeur."""
        labels = torch.full((n,), int(class_label), dtype=torch.long, device=self.device)
        _, condition = self.module._prepare_condition_maps(labels, n, self.device)
        return condition

    @torch.no_grad()
    def encode(self, x: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            labels = torch.full((x.size(0),), int(class_label), dtype=torch.long, device=self.device)
            mu, _ = self.module.encode(x, labels)
        else:
            mu, _ = self.module.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            return self.module.decode(z, self._condition_vector(class_label, z.size(0)))
        return self.module.decode(z)


def load(
    checkpoint_path: Path,
    device: torch.device,
    config_path: Optional[Path] = None,
    **_ignored,
) -> SylvainMnistAdapter:
    """Charge un checkpoint de `src/` a partir de son YAML de configuration.

    Contrairement aux checkpoints de `projects/`, ceux-ci ne contiennent pas
    leur propre configuration : l'architecture est decrite par le YAML, qui est
    donc obligatoire.
    """
    if config_path is None:
        raise ValueError("Les modeles de src/ requierent un 'config_path'.")

    config = load_yaml_config(str(config_path))
    dataset = config["dataset"]
    num_classes = dataset["num_classes"]

    # `build_dataloaders` telechargerait MNIST et construirait trois loaders
    # dont l'API n'a aucun usage : on fabrique le DatasetInfo depuis le YAML.
    dataset_info = DatasetInfo(
        image_size=tuple(dataset["image_size"]),
        channels=dataset["channels"],
        num_conditions=num_classes,
        class_names=[str(index) for index in range(num_classes)],
        condition_mode=dataset.get("condition_mode", "one_hot"),
    )

    module = build_model_from_config(config, dataset_info)
    module = load_checkpoint(module, str(checkpoint_path), device)
    return SylvainMnistAdapter(module, device)
