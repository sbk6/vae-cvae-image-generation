"""Interpolation linéaire dans l'espace latent entre deux exemples réels.

On encode deux images de classes différentes (mu déterministe, sans le bruit
de la reparamétrisation), on interpole linéairement entre les deux vecteurs
latents, puis on décode chaque point intermédiaire. Si le VAE a bien appris
un espace latent continu, la transition doit être visuellement progressive.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


@torch.no_grad()
def interpolate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    class_a: int = 3,
    class_b: int = 8,
    steps: int = 10,
) -> None:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    image_a, image_b = None, None
    for x, y in test_loader:
        for i in range(x.size(0)):
            if image_a is None and y[i].item() == class_a:
                image_a = x[i:i + 1]
            if image_b is None and y[i].item() == class_b:
                image_b = x[i:i + 1]
        if image_a is not None and image_b is not None:
            break

    image_a, image_b = image_a.to(device), image_b.to(device)
    if conditioned:
        mu_a, _ = model.encode(image_a, torch.tensor([class_a], device=device))
        mu_b, _ = model.encode(image_b, torch.tensor([class_b], device=device))
    else:
        mu_a, _ = model.encode(image_a)
        mu_b, _ = model.encode(image_b)

    alphas = torch.linspace(0, 1, steps, device=device)
    frames = []
    for alpha in alphas:
        z = (1 - alpha) * mu_a + alpha * mu_b
        if conditioned:
            # condition fixée sur la classe de départ pour observer la morphologie du chiffre
            _, c_vec = model._prepare_condition_maps(torch.tensor([class_a], device=device), 1, device)
            x_hat = model.decode(z, c_vec)
        else:
            x_hat = model.decode(z)
        frames.append(x_hat.cpu())

    frames = torch.cat(frames, dim=0)
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 1.6))
    for i, ax in enumerate(axes):
        img = (frames[i, 0] + 1) / 2  # de [-1,1] vers [0,1]
        ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes[0].set_title(str(class_a), fontsize=10)
    axes[-1].set_title(str(class_b), fontsize=10)
    fig.suptitle(f"Interpolation dans l'espace latent : {class_a} -> {class_b}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--class-a", type=int, default=3)
    parser.add_argument("--class-b", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    interpolate(args.config, args.checkpoint, args.output, args.class_a, args.class_b, args.steps)
