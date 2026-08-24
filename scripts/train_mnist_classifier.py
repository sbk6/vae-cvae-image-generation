"""Entraîne un classifieur MNIST dédié, pour mesurer la contrôlabilité du CVAE
sans le biais du proxy plus-proche-centroïde (même principe que le classifieur
Fashion-MNIST de David). Rapide sur CPU : MNIST est la tâche de classification
la plus simple du projet.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

from src.data.datasets import build_dataloaders
from src.models.classifier import DigitClassifier
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mnist_vae.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/experiments/mnist_classifier/best_checkpoint.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_yaml_config(args.config)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = DigitClassifier(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        num_classes=dataset_info.num_conditions,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        print(f"epoch {epoch}: val_accuracy={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

    model.load_state_dict(torch.load(args.output, map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"test_accuracy={test_acc:.4f}")

    summary_path = Path(args.output).parent / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_accuracy": best_val_acc, "test_accuracy": test_acc, "epochs": args.epochs, "seed": args.seed}, f, indent=2)
    print(f"Checkpoint : {args.output}")
    print(f"Résumé : {summary_path}")


if __name__ == "__main__":
    main()
