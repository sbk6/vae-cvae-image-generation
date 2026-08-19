"""Projection 2D (t-SNE) de l'espace latent, colorée par classe.

Sert à vérifier visuellement si le VAE/CVAE organise l'espace latent par
classe de chiffre, même si la classe n'est jamais donnée explicitement au
VAE (elle l'est en entrée pour le CVAE).
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


@torch.no_grad()
def collect_latents(model, loader, device, conditioned: bool, max_points: int):
    mus, labels = [], []
    total = 0
    for x, y in loader:
        x = x.to(device)
        if conditioned:
            mu, _ = model.encode(x, y.to(device))
        else:
            mu, _ = model.encode(x)
        mus.append(mu.cpu())
        labels.append(y)
        total += x.size(0)
        if total >= max_points:
            break
    return torch.cat(mus)[:max_points], torch.cat(labels)[:max_points]


def plot_latent_tsne(config_path: str, checkpoint_path: str, output_path: str, max_points: int = 2000) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    mus, labels = collect_latents(model, test_loader, device, conditioned, max_points)

    print(f"Projection t-SNE de {mus.shape[0]} points en dimension {mus.shape[1]} -> 2D ...")
    embedding = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(mus.numpy())

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels.numpy(), cmap="tab10", s=8, alpha=0.8)
    legend = ax.legend(*scatter.legend_elements(), title="classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.add_artist(legend)
    model_name = "CVAE" if conditioned else "VAE"
    ax.set_title(f"Espace latent {model_name} projeté en 2D (t-SNE), coloré par classe réelle")
    ax.set_xlabel("dim. t-SNE 1")
    ax.set_ylabel("dim. t-SNE 2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-points", type=int, default=2000)
    args = parser.parse_args()
    plot_latent_tsne(args.config, args.checkpoint, args.output, args.max_points)
