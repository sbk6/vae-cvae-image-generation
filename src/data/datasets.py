import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    image_size: Tuple[int, int]
    channels: int
    num_conditions: int
    class_names: List[str]
    condition_mode: str


def build_mnist_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    image_size = tuple(config["dataset"]["image_size"])
    channels = config["dataset"]["channels"]
    condition_mode = config["dataset"].get("condition_mode", "one_hot")
    num_classes = config["dataset"]["num_classes"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    val_size = int(0.1 * len(train_set))
    train_size = len(train_set) - val_size
    train_subset, val_subset = random_split(train_set, [train_size, val_size])

    # Sous-échantillonnage optionnel du train set : utile pour accélérer les
    # études d'ablation sur CPU sans changer le protocole (val/test restent complets).
    max_train_samples = config["dataset"].get("train_subset")
    if max_train_samples is not None and max_train_samples < len(train_subset):
        train_subset = torch.utils.data.Subset(train_subset, list(range(max_train_samples)))

    batch_size = config["training"]["batch_size"]
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = [str(i) for i in range(num_classes)]
    dataset_info = DatasetInfo(
        image_size=image_size,
        channels=channels,
        num_conditions=num_classes if condition_mode == "one_hot" else len(class_names),
        class_names=class_names,
        condition_mode=condition_mode,
    )

    return train_loader, val_loader, test_loader, dataset_info


def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    dataset_name = config["dataset"]["name"].lower()
    if dataset_name == "mnist":
        return build_mnist_loaders(config)
    if dataset_name == "fashion_mnist":
        # Point d'extension : remplacer datasets.MNIST par datasets.FashionMNIST
        # et éventuellement adapter les noms de classes si nécessaire.
        return build_mnist_loaders(config)
    if dataset_name == "celeba":
        # Point d'extension : charger le dataset CelebA, gérer la normalisation couleur,
        # et construire les conditions multi_label à partir des attributs demandés.
        raise NotImplementedError("CelebA loader is not encore implémente")

    raise ValueError(f"Dataset inconnu : {dataset_name}")
