import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.datasets import build_dataloaders
from src.utils.config import load_yaml_config


def save_real_image_grid(config_path: str, output_path: str) -> None:
    config_dict = load_yaml_config(config_path)
    train_loader, _, _, dataset_info = build_dataloaders(config_dict)
    batch = next(iter(train_loader))
    x, y = batch
    grid = make_grid(x[:16], nrow=4, normalize=True, value_range=(-1.0, 1.0))
    figure = plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title("Exemples reels MNIST")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved real image grid to {output_path}")


if __name__ == "__main__":
    config_path = os.environ.get("CONFIG_PATH", "configs/mnist_vae.yaml")
    output_path = os.environ.get("OUTPUT_PATH", "reports/figures/mnist_real_grid.png")
    save_real_image_grid(config_path, output_path)
