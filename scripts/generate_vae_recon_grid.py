"""Génère une grille comparant images réelles (ligne du haut) et
leur reconstruction par le VAE (ligne du bas), plus une grille d'échantillons
purement générés (z ~ N(0, I)), pour juger la génération "libre" du VAE.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from torchvision.utils import make_grid, save_image

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


@torch.no_grad()
def generate(config_path: str, checkpoint_path: str, recon_output: str, sample_output: str, n: int = 8) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    x, _ = next(iter(test_loader))
    x = x[:n].to(device)
    mu, logvar = model.encode(x)
    x_hat = model.decode(mu)

    comparison = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    grid = make_grid(comparison, nrow=n, normalize=True, value_range=(-1.0, 1.0))
    Path(recon_output).parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, recon_output)
    print(f"Saved reconstruction grid (haut=réel, bas=reconstruit) to {recon_output}")

    z = torch.randn(n * n, model.latent_dim, device=device)
    samples = model.decode(z).cpu()
    grid_samples = make_grid(samples, nrow=n, normalize=True, value_range=(-1.0, 1.0))
    save_image(grid_samples, sample_output)
    print(f"Saved unconditional sample grid to {sample_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_vae.yaml")
    parser.add_argument("--checkpoint", type=str, default="reports/experiments/vae_main/best_checkpoint.pth")
    parser.add_argument("--recon-output", type=str, default="reports/figures/vae_reconstruction_grid.png")
    parser.add_argument("--sample-output", type=str, default="reports/figures/vae_random_samples_grid.png")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    generate(args.config, args.checkpoint, args.recon_output, args.sample_output, args.n)
