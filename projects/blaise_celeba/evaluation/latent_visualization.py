"""Projection 2D (t-SNE) de l'espace latent CelebA, coloree par un attribut.

Difference importante avec MNIST/Fashion-MNIST : ces datasets ont des classes
exclusives (un chiffre est soit un 3, soit un 8, jamais les deux), donc une
seule couleur par point suffit a lire le graphe. CelebA n'a que des attributs
binaires non exclusifs (un visage peut etre "Smiling" et "Male" en meme
temps) : il n'existe pas de "classe" unique a colorer. On choisit donc un
attribut a la fois pour la couleur (par defaut "Smiling", le plus visible sur
une image de visage), et on le precise explicitement dans le titre de la
figure pour ne pas laisser croire a une classe exclusive.

Usage (depuis projects/blaise_celeba/) :
    python -m evaluation.latent_visualization --config configs/celeba_vae.yaml \\
        --checkpoint results/experiments/vae_main/best_checkpoint.pth \\
        --output results/figures/latent_tsne_vae.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from data.dataset import build_dataloaders
from models.common import build_model_from_config, load_checkpoint
from training.trainer import get_device
from utils import load_yaml_config


@torch.no_grad()
def collect_latents(model, loader, device, conditioned: bool, max_points: int):
    mus, attrs_all = [], []
    total = 0
    for x, attrs in loader:
        x = x.to(device)
        if conditioned:
            mu, _ = model.encode(x, attrs.to(device).float())
        else:
            mu, _ = model.encode(x)
        mus.append(mu.cpu())
        attrs_all.append(attrs)
        total += x.size(0)
        if total >= max_points:
            break
    return torch.cat(mus)[:max_points], torch.cat(attrs_all)[:max_points]


def plot_latent_tsne(config_path: str, checkpoint_path: str, output_path: str, color_attribute: str, max_points: int) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    mus, attrs = collect_latents(model, test_loader, device, conditioned, max_points)

    if color_attribute not in dataset_info.attribute_names:
        raise ValueError(
            f"'{color_attribute}' n'est pas un attribut de conditionnement de ce run "
            f"({dataset_info.attribute_names}). Choisir --color-attribute parmi cette liste."
        )
    attribute_index = dataset_info.attribute_names.index(color_attribute)
    colors = attrs[:, attribute_index].numpy()

    print(f"Projection t-SNE de {mus.shape[0]} points en dimension {mus.shape[1]} -> 2D ...")
    embedding = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(mus.numpy())

    # Deux appels scatter distincts (plutot qu'un seul colore par une valeur
    # continue) : la legende categorielle qui en resulte reste lisible et
    # entierement a l'interieur de la figure, contrairement a
    # `legend_elements()` sur des couleurs continues, dont la boite de
    # legende externe se retrouvait tronquee a la sauvegarde.
    fig, ax = plt.subplots(figsize=(7, 6))
    is_active = colors > 0.5
    ax.scatter(embedding[~is_active, 0], embedding[~is_active, 1], c="tab:blue", s=8, alpha=0.8, label=f"{color_attribute} = 0")
    ax.scatter(embedding[is_active, 0], embedding[is_active, 1], c="tab:red", s=8, alpha=0.8, label=f"{color_attribute} = 1")
    ax.legend(loc="best")
    model_name = "CVAE" if conditioned else "VAE"
    ax.set_title(f"Espace latent {model_name} (CelebA) projete en 2D (t-SNE), colore par '{color_attribute}'")
    ax.set_xlabel("dim. t-SNE 1")
    ax.set_ylabel("dim. t-SNE 2")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardee dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--color-attribute", type=str, default="Smiling")
    parser.add_argument("--max-points", type=int, default=1500)
    args = parser.parse_args()
    plot_latent_tsne(args.config, args.checkpoint, args.output, args.color_attribute, args.max_points)
