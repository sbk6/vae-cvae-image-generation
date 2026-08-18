"""Genere les grilles d'images qualitatives : reconstruction, echantillons
libres du VAE, et echantillons conditionnes du CVAE (une ligne par
combinaison d'attributs).

Ne reentraine jamais le modele : charge un checkpoint deja entraine.

Usage (depuis projects/blaise_celeba/) :
    python -m evaluation.generate_grids --model vae \\
        --config configs/celeba_vae.yaml \\
        --checkpoint results/experiments/vae_main/best_checkpoint.pth

    python -m evaluation.generate_grids --model cvae \\
        --config configs/celeba_cvae.yaml \\
        --checkpoint results/experiments/cvae_main/best_checkpoint.pth
"""
import argparse
import itertools
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from data.dataset import build_dataloaders
from models.common import build_model_from_config, load_checkpoint
from training.trainer import get_device
from utils import load_yaml_config


@torch.no_grad()
def generate_vae_grids(config_path: str, checkpoint_path: str, output_dir: str, n: int = 8) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    x, _ = next(iter(test_loader))
    x = x[:n].to(device)
    mu, _ = model.encode(x)
    x_hat = model.decode(mu)

    comparison = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    grid = make_grid(comparison, nrow=n, normalize=True, value_range=(-1.0, 1.0))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recon_path = output_dir / "vae_reconstruction_grid.png"
    save_image(grid, recon_path)
    print(f"Grille reconstruction (haut=reel, bas=reconstruit) sauvegardee dans {recon_path}")

    z = torch.randn(n * n, model.latent_dim, device=device)
    samples = model.decode(z).cpu()
    grid_samples = make_grid(samples, nrow=n, normalize=True, value_range=(-1.0, 1.0))
    sample_path = output_dir / "vae_random_samples_grid.png"
    save_image(grid_samples, sample_path)
    print(f"Grille d'echantillons libres sauvegardee dans {sample_path}")


@torch.no_grad()
def generate_cvae_grid(config_path: str, checkpoint_path: str, output_dir: str, samples_per_combo: int = 8) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    num_attrs = dataset_info.num_conditions
    combos = list(itertools.product([0.0, 1.0], repeat=num_attrs))

    images = []
    for combo in combos:
        condition = torch.tensor(combo, dtype=torch.float32)
        samples = model.sample(condition, n=samples_per_combo).cpu()
        images.extend(samples[i] for i in range(samples_per_combo))

    grid = make_grid(images, nrow=samples_per_combo, normalize=True, value_range=(-1.0, 1.0))
    output_path = Path(output_dir) / "cvae_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)

    legend = "\n".join(
        f"ligne {i}: " + ", ".join(f"{name}={int(value)}" for name, value in zip(dataset_info.attribute_names, combo))
        for i, combo in enumerate(combos)
    )
    print(f"Grille CVAE (une ligne par combinaison d'attributs) sauvegardee dans {output_path}\n{legend}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vae", "cvae"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    if args.model == "vae":
        generate_vae_grids(args.config, args.checkpoint, args.output_dir, args.n)
    else:
        generate_cvae_grid(args.config, args.checkpoint, args.output_dir, args.n)


if __name__ == "__main__":
    main()
