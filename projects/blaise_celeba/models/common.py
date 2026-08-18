"""Construction de modele depuis une config, et chargement de checkpoint.

Reutilise par tous les scripts d'evaluation/visualisation pour eviter de
dupliquer la logique de reconstruction du modele a partir du YAML.
"""
from typing import Dict

import torch

# Import relatif : voir la note dans models/vae.py.
from .cvae import CVAE
from .vae import VAE


def build_model_from_config(config: Dict, dataset_info) -> torch.nn.Module:
    model_type = config.get("model", {}).get("type", "vae")
    if model_type == "cvae":
        return CVAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config["model"]["latent_dim"],
            num_conditions=dataset_info.num_conditions,
            hidden_channels=config["model"]["hidden_channels"],
        )
    return VAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=config["model"]["latent_dim"],
        hidden_channels=config["model"]["hidden_channels"],
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
