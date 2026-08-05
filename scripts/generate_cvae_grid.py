import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

from src.data.datasets import build_dataloaders
from src.models.cvae import CVAE
from src.training.trainer import train
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed


def generate_grid(config_path: str, output_path: str, samples_per_class: int = 8) -> None:
    config = load_yaml_config(config_path)
    set_seed(config.get("training", {}).get("seed", 42))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = CVAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=config.get("model", {}).get("latent_dim", 16),
        num_conditions=dataset_info.num_conditions,
        condition_mode=dataset_info.condition_mode,
        hidden_channels=config.get("model", {}).get("hidden_channels", 32),
    )

    # quick train (smoke-test behavior enforced by train)
    config["smoke_test"] = True
    train(config, model, train_loader, val_loader)

    # load best checkpoint if present
    ckpt_path = Path(config.get("training", {}).get("output_dir", "reports")) / "best_checkpoint.pth"
    if ckpt_path.exists():
        state = model.state_dict()
        ckpt = torch.load(ckpt_path, map_location=next(model.parameters()).device)
        model.load_state_dict(ckpt["model_state_dict"])

    device = next(model.parameters()).device
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
    plt.title("CVAE conditioned samples")
    plt.imshow(grid.permute(1, 2, 0), cmap="gray")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved CVAE sample grid to {output_path}")


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_cvae.yaml")
    parser.add_argument("--output", type=str, default="reports/figures/cvae_grid.png")
    parser.add_argument("--samples-per-class", type=int, default=8)
    args = parser.parse_args()
    generate_grid(args.config, args.output, samples_per_class=args.samples_per_class)
