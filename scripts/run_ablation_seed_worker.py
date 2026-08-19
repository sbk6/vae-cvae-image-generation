"""Entraîne UN VAE pour UNE combinaison (beta, seed) de l'étude d'ablation multi-seed.

Ce script est conçu pour être lancé plusieurs fois en parallèle (un processus
par combinaison beta/seed), afin de paralléliser l'étude d'ablation sur CPU.
Voir `scripts/run_ablation_seeds.py` pour l'orchestration complète et
`scripts/aggregate_ablation_seeds.py` pour l'agrégation des résultats.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import build_dataloaders
from src.models.vae import VAE
from src.training.trainer import train
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_beta.yaml")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config["training"]["beta"] = args.beta
    config["training"]["seed"] = args.seed
    config["training"]["output_dir"] = str(Path(config["training"]["output_dir"]) / f"beta_{args.beta}_seed_{args.seed}")
    config["smoke_test"] = False

    if config["training"].get("mlflow", {}).get("enabled", False):
        config["training"]["mlflow"]["run_name"] = f"ablation_beta_{args.beta}_seed_{args.seed}"
        config["training"]["mlflow"].setdefault("tags", {})
        config["training"]["mlflow"]["tags"]["beta"] = str(args.beta)
        config["training"]["mlflow"]["tags"]["seed"] = str(args.seed)
        config["training"]["mlflow"]["tags"]["phase"] = "ablation_seeds"

    set_seed(args.seed)
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = VAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=config["model"]["latent_dim"],
        hidden_channels=config["model"]["hidden_channels"],
    )

    print(f"=== beta={args.beta} seed={args.seed} -> {config['training']['output_dir']} ===")
    train(config, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
