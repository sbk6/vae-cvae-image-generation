"""Entraîne UN modèle principal (VAE ou CVAE) pour UN seed donné, sur les
données complètes, à 20 epochs — variante multi-seed des modèles principaux
(`vae_main` / `cvae_main`), pour vérifier leur robustesse comme on l'a fait
pour l'étude d'ablation sur beta.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import build_dataloaders
from src.models.cvae import CVAE
from src.models.vae import VAE
from src.training.trainer import train
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["vae", "cvae"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    config_path = "configs/mnist_vae.yaml" if args.model == "vae" else "configs/mnist_cvae.yaml"
    config = load_yaml_config(config_path)
    config["training"]["seed"] = args.seed
    config["training"]["epochs"] = args.epochs
    config["training"]["output_dir"] = f"reports/experiments/{args.model}_seeds/seed_{args.seed}"
    config["smoke_test"] = False

    if config["training"].get("mlflow", {}).get("enabled", False):
        config["training"]["mlflow"]["run_name"] = f"{args.model}_main_seed_{args.seed}"
        config["training"]["mlflow"].setdefault("tags", {})
        config["training"]["mlflow"]["tags"]["seed"] = str(args.seed)
        config["training"]["mlflow"]["tags"]["phase"] = "main_seeds"
        config["training"]["mlflow"]["tags"]["model"] = args.model

    set_seed(args.seed)
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    if args.model == "cvae":
        model = CVAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config["model"]["latent_dim"],
            num_conditions=dataset_info.num_conditions,
            condition_mode=dataset_info.condition_mode,
            hidden_channels=config["model"]["hidden_channels"],
        )
    else:
        model = VAE(
            channels=dataset_info.channels,
            image_size=dataset_info.image_size,
            latent_dim=config["model"]["latent_dim"],
            hidden_channels=config["model"]["hidden_channels"],
        )

    print(f"=== {args.model} seed={args.seed} epochs={args.epochs} -> {config['training']['output_dir']} ===")
    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
