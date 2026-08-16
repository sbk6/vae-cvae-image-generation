"""Génère une grille d'images échantillonnées par le CVAE, une ligne par classe.

Contrairement à la version précédente, ce script NE réentraîne PAS le modèle :
il charge un checkpoint déjà entraîné (voir `python -m src.training.train`).
Cela évite d'écraser un modèle bien entraîné par un entraînement express.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


def generate_grid(config_path: str, checkpoint_path: str, output_path: str, samples_per_class: int = 8) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    rows = dataset_info.num_conditions
    cols = samples_per_class
    images = []
    for cls in range(dataset_info.num_conditions):
        samples = model.sample(cls, n=cols).cpu()
        for i in range(cols):
            images.append(samples[i])

    grid = make_grid(images, nrow=cols, normalize=True, value_range=(-1.0, 1.0))
    figure = plt.figure(figsize=(cols, rows))
    plt.axis("off")
    plt.title("CVAE — échantillons conditionnés (une ligne par classe 0..9)")
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved CVAE sample grid to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_cvae.yaml")
    parser.add_argument("--checkpoint", type=str, default="reports/experiments/cvae_main/best_checkpoint.pth")
    parser.add_argument("--output", type=str, default="reports/figures/cvae_grid.png")
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args()
    generate_grid(args.config, args.checkpoint, args.output, samples_per_class=args.samples_per_class)
