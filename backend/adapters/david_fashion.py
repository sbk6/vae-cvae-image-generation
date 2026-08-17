"""Adaptateur pour les modeles Fashion-MNIST de `projects/david_fashion_mnist/`.

Specificites de cette famille :
- donnees en [0, 1] (transforms.ToTensor() seul, pas de Normalize) ;
- decodeur termine par Sigmoid, donc sortie dans [0, 1] ;
- loss de reconstruction en BCE et non en MSE ;
- CVAE conditionne par concatenation d'un vecteur one-hot apres l'aplatissement
  des cartes de convolution, et non par des canaux spatiaux ;
- `forward()` renvoie 4 valeurs (reconstruction, mu, logvar, z) ;
- pas de YAML : l'architecture est decrite dans le checkpoint lui-meme.

L'architecture est deduite directement des formes du `state_dict` plutot que
de la cle `configuration` du checkpoint. Les poids sont la seule source de
verite : un checkpoint dont les metadonnees auraient derive se chargera quand
meme correctement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from backend.adapters.base import ModelAdapter
from projects.david_fashion_mnist.models.cvae import CVAE
from projects.david_fashion_mnist.models.vae import VAE

# 64 canaux x 7 x 7 apres les deux convolutions stride 2 sur du 28x28.
ENCODER_OUTPUT_DIM = 64 * 7 * 7


class DavidFashionAdapter(ModelAdapter):
    """Enveloppe un VAE ou CVAE de `projects/david_fashion_mnist/models/`."""

    @property
    def latent_dim(self) -> int:
        return int(self.module.latent_dim)

    @property
    def is_conditional(self) -> bool:
        return isinstance(self.module, CVAE)

    @property
    def num_conditions(self) -> int:
        return int(getattr(self.module, "num_classes", 0) or 0)

    @property
    def output_range(self) -> Tuple[float, float]:
        # Sigmoid en sortie de decodeur.
        return (0.0, 1.0)

    @property
    def input_range(self) -> Tuple[float, float]:
        # ToTensor() seul : pas de recentrage autour de zero.
        return (0.0, 1.0)

    def _labels(self, class_label: int, n: int) -> torch.Tensor:
        """Ses modeles attendent un tenseur d'indices int64 de forme (n,)."""
        return torch.full((n,), int(class_label), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            mu, _ = self.module.encode(x, self._labels(class_label, x.size(0)))
        else:
            mu, _ = self.module.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor, class_label: Optional[int] = None) -> torch.Tensor:
        if self.is_conditional:
            if class_label is None:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            return self.module.decode(z, self._labels(class_label, z.size(0)))
        return self.module.decode(z)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Recupere le state_dict, que le checkpoint soit enveloppe ou non."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        # Checkpoint sauvegarde directement en state_dict brut.
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Format de checkpoint non reconnu : 'model_state_dict' introuvable.")


def _infer_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Deduit latent_dim, hidden_dim et num_classes des formes du state_dict.

    - fc_mu.weight       : (latent_dim, hidden_dim)
    - encoder_hidden.0.weight : (hidden_dim, ENCODER_OUTPUT_DIM [+ num_classes])
    """
    if "fc_mu.weight" not in state_dict:
        raise ValueError("Checkpoint invalide : 'fc_mu.weight' introuvable.")

    latent_dim, hidden_dim = state_dict["fc_mu.weight"].shape

    conditional = "encoder_convolutions.0.weight" in state_dict
    num_classes = 0
    if conditional:
        encoder_input = state_dict["encoder_hidden.0.weight"].shape[1]
        num_classes = int(encoder_input) - ENCODER_OUTPUT_DIM
        if num_classes <= 1:
            raise ValueError(
                f"Nombre de classes deduit incoherent ({num_classes}) : "
                "le checkpoint ne correspond pas a l'architecture attendue."
            )

    return {
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "num_classes": num_classes,
        "conditional": conditional,
    }


def _read_checkpoint(checkpoint_path: Path, map_location) -> Any:
    """Lit un checkpoint en desactivant l'execution de pickle arbitraire.

    `torch.load` deserialise par defaut via pickle, ce qui execute du code
    contenu dans le fichier. Les checkpoints de David n'embarquent que des
    tenseurs et des types primitifs, donc `weights_only=True` suffit et evite
    d'accorder une confiance inutile a un fichier recu de l'exterieur.

    Le repli couvre les checkpoints plus anciens qui contiendraient des objets
    Python serialises ; il n'est pas atteint par les poids actuels.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load(
    checkpoint_path: Path,
    device: torch.device,
    **_ignored,
) -> DavidFashionAdapter:
    """Charge un checkpoint Fashion-MNIST.

    Aucun YAML n'est requis : l'architecture est reconstruite depuis les poids.
    """
    checkpoint = _read_checkpoint(checkpoint_path, device)
    state_dict = _extract_state_dict(checkpoint)
    architecture = _infer_architecture(state_dict)

    if architecture["conditional"]:
        module = CVAE(
            latent_dim=architecture["latent_dim"],
            hidden_dim=architecture["hidden_dim"],
            num_classes=architecture["num_classes"],
        )
    else:
        module = VAE(
            latent_dim=architecture["latent_dim"],
            hidden_dim=architecture["hidden_dim"],
        )

    module.load_state_dict(state_dict)
    return DavidFashionAdapter(module, device)


def describe(checkpoint_path: Path) -> Dict[str, Any]:
    """Lit les metadonnees d'un checkpoint sans instancier le modele.

    Sert au catalogue pour afficher beta et le type de modele sans payer le
    cout d'un chargement complet.
    """
    checkpoint = _read_checkpoint(checkpoint_path, "cpu")
    if not isinstance(checkpoint, dict):
        return {}
    configuration = checkpoint.get("configuration") or {}
    return {
        "model_type": checkpoint.get("model_type"),
        "beta": checkpoint.get("beta", configuration.get("beta")),
        "epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "dataset": configuration.get("dataset"),
    }
