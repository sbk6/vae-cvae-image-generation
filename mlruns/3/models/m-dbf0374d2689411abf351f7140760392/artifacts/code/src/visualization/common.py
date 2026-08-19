from pathlib import Path
from typing import Dict, Tuple

import torch

from src.models.cvae import CVAE
from src.models.vae import VAE


def build_model_from_config(config: Dict, dataset_info) -> torch.nn.Module:
    model_type = config.get("model", {}).get("type", "vae")
    if model_type == "cvae":
        return CVAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config["model"]["latent_dim"],
            num_conditions=dataset_info.num_conditions,
            condition_mode=dataset_info.condition_mode,
            hidden_channels=config["model"]["hidden_channels"],
        )
    return VAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=config["model"]["latent_dim"],
        hidden_channels=config["model"]["hidden_channels"],
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model
