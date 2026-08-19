import argparse
import os
from pathlib import Path
from typing import Dict

from src.data.datasets import build_dataloaders
from src.models.cvae import CVAE
from src.models.vae import VAE
from src.training.trainer import train
from src.utils.config import load_yaml_config, merge_args
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement d'un VAE/CVAE")
    parser.add_argument("--config", type=str, required=True, help="Fichier de configuration YAML")
    parser.add_argument("--seed", type=int, help="Seed de reproductibilite")
    parser.add_argument("--batch-size", type=int, help="Taille de batch")
    parser.add_argument("--lr", type=float, help="Taux d'apprentissage")
    parser.add_argument("--epochs", type=int, help="Nombre d'epochs")
    parser.add_argument("--beta", type=float, help="Coefficient beta")
    parser.add_argument("--device", type=str, help="Device: cpu ou cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Execute un run court de verification")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict:
    config = load_yaml_config(args.config)
    merged = merge_args(config, args)
    if merged.get("seed") is not None:
        merged["training"]["seed"] = merged["seed"]
    return merged


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_seed(config["training"].get("seed", 42))

    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model_type = config.get("model", {}).get("type", "vae")
    if model_type == "cvae":
        model = CVAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config.get("model", {}).get("latent_dim", 16),
            num_conditions=dataset_info.num_conditions,
            condition_mode=dataset_info.condition_mode,
            hidden_channels=config.get("model", {}).get("hidden_channels", 32),
        )
    else:
        model = VAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config.get("model", {}).get("latent_dim", 16),
            hidden_channels=config.get("model", {}).get("hidden_channels", 32),
        )

    config["smoke_test"] = args.smoke_test
    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
