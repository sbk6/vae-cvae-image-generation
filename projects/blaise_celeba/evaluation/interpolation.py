"""Interpolation lineaire dans l'espace latent entre deux vrais visages de test.

On encode deux images (mu deterministe, sans le bruit de la reparametrisation),
on interpole lineairement entre les deux vecteurs latents, puis on decode
chaque point intermediaire. Si le VAE/CVAE a bien appris un espace latent
continu, la transition doit etre visuellement progressive (pas de saut brutal
au milieu), signe indirect que le terme KL a bien regularise l'espace latent.

Contrairement a l'interpolation MNIST (entre deux classes de chiffres,
"3" et "8"), CelebA n'a pas de classes discretes : les deux images sont donc
choisies par leur index dans le test set plutot que par une classe.

Usage (depuis projects/blaise_celeba/) :
    python -m evaluation.interpolation --config configs/celeba_vae.yaml \\
        --checkpoint results/experiments/vae_main/best_checkpoint.pth \\
        --output results/figures/interpolation_vae_0_to_1.png \\
        --index-a 0 --index-b 1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import build_dataloaders
from models.common import build_model_from_config, load_checkpoint
from training.trainer import get_device
from utils import load_yaml_config


def _to_display(image: torch.Tensor) -> np.ndarray:
    """[-1, 1] CHW -> [0, 1] HWC, pour matplotlib."""
    array = (image.permute(1, 2, 0).numpy() + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0)


@torch.no_grad()
def interpolate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    index_a: int,
    index_b: int,
    steps: int = 10,
) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    test_set = test_loader.dataset
    image_a, attrs_a = test_set[index_a]
    image_b, _ = test_set[index_b]
    image_a = image_a.unsqueeze(0).to(device)
    image_b = image_b.unsqueeze(0).to(device)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    if conditioned:
        c_a = attrs_a.unsqueeze(0).to(device).float()
        mu_a, _ = model.encode(image_a, c_a)
        mu_b, _ = model.encode(image_b, c_a)  # meme condition pour isoler l'effet du latent
    else:
        mu_a, _ = model.encode(image_a)
        mu_b, _ = model.encode(image_b)

    alphas = torch.linspace(0, 1, steps, device=device)
    frames = []
    for alpha in alphas:
        z = (1 - alpha) * mu_a + alpha * mu_b
        x_hat = model.decode(z, c_a) if conditioned else model.decode(z)
        frames.append(x_hat.cpu())
    frames = torch.cat(frames, dim=0)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.4, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(_to_display(frames[i]))
        ax.axis("off")
    axes[0].set_title(f"#{index_a}", fontsize=10)
    axes[-1].set_title(f"#{index_b}", fontsize=10)
    model_name = "CVAE" if conditioned else "VAE"
    fig.suptitle(f"{model_name} CelebA : interpolation dans l'espace latent (test #{index_a} -> #{index_b})")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardee dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--index-a", type=int, default=0)
    parser.add_argument("--index-b", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    interpolate(args.config, args.checkpoint, args.output, args.index_a, args.index_b, args.steps)
