"""Point d'entree CLI pour entrainer le VAE ou le CVAE sur CelebA.

Usage (depuis projects/blaise_celeba/, environnement virtuel active) :

    python -m training.train --config configs/celeba_vae.yaml
    python -m training.train --config configs/celeba_cvae.yaml
    python -m training.train --config configs/celeba_vae.yaml --smoke-test
"""
import argparse

from data.dataset import build_dataloaders
from models.cvae import CVAE
from models.vae import VAE
from training.trainer import train
from utils import load_yaml_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement d'un VAE/CVAE sur CelebA")
    parser.add_argument("--config", type=str, required=True, help="Fichier de configuration YAML")
    parser.add_argument("--beta", type=float, help="Ecrase le beta du YAML (utile pour l'ablation)")
    parser.add_argument("--output-dir", type=str, help="Ecrase le dossier de sortie du YAML")
    parser.add_argument("--epochs", type=int, help="Ecrase le nombre d'epochs du YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run court de verification (1 epoch, 5 batches)")
    return parser.parse_args()


def build_model(config: dict, dataset_info):
    model_type = config.get("model", {}).get("type", "vae")
    latent_dim = config["model"].get("latent_dim", 64)
    hidden_channels = config["model"].get("hidden_channels", 32)

    if model_type == "cvae":
        return CVAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=latent_dim,
            num_conditions=dataset_info.num_conditions,
            hidden_channels=hidden_channels,
        )
    return VAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.beta is not None:
        config["training"]["beta"] = args.beta
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    config["smoke_test"] = args.smoke_test

    set_seed(config["training"].get("seed", 42))

    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)
    model = build_model(config, dataset_info)

    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
