"""Précalcule, pour un VAE entraîné, un point représentatif de l'espace latent
par classe (la moyenne des `mu` de toutes les images de test de cette classe).

Sert à l'endpoint d'interpolation déployé : plutôt que d'avoir à charger le
test set complet dans le serveur de production pour aller chercher "un
exemple de la classe 3", on part d'un point latent stable et représentatif,
précalculé une fois pour toutes.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


@torch.no_grad()
def compute_centroids(config_path: str, checkpoint_path: str, output_path: str) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    sums = defaultdict(lambda: torch.zeros(config["model"]["latent_dim"]))
    counts = defaultdict(int)

    for x, y in test_loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        for i in range(x.size(0)):
            label = int(y[i].item())
            sums[label] += mu[i].cpu()
            counts[label] += 1

    centroids = {str(label): (sums[label] / counts[label]).tolist() for label in sorted(counts)}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(centroids, f, indent=2)
    print(f"Centroïdes latents sauvegardés dans {output_path} ({len(centroids)} classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_vae.yaml")
    parser.add_argument("--checkpoint", type=str, default="reports/experiments/vae_main/best_checkpoint.pth")
    parser.add_argument("--output", type=str, default="reports/experiments/vae_main/latent_centroids.json")
    args = parser.parse_args()
    compute_centroids(args.config, args.checkpoint, args.output)
