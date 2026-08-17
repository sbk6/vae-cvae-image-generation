"""Precalcule des echantillons d'images reelles pour la demo web.

Pourquoi : la demo a besoin de vraies images (reconstruction, interpolation),
mais embarquer torchvision + les datasets complets dans l'image Docker
couterait ~64 Mo par dataset et un telechargement au premier demarrage. On
extrait donc une fois pour toutes quelques images par classe dans des .npz de
~20 Ko, versionnes avec le code.

Les images sont stockees en uint8 [0, 255], **sans normalisation** : MNIST et
Fashion-MNIST sont consommes par deux familles de modeles qui n'attendent pas
la meme plage ([-1, 1] contre [0, 1]). La normalisation est appliquee cote API
par l'adaptateur du modele cible.

Usage :
    python scripts/build_demo_fixtures.py                  # les deux datasets
    python scripts/build_demo_fixtures.py --dataset mnist
    python scripts/build_demo_fixtures.py --per-class 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from torchvision import datasets

ASSETS_DIR = ROOT_DIR / "backend" / "assets"

DATASETS = {
    "mnist": {
        "loader": datasets.MNIST,
        "filename": "mnist_samples.npz",
        "label": "MNIST",
    },
    "fashion_mnist": {
        "loader": datasets.FashionMNIST,
        "filename": "fashion_mnist_samples.npz",
        "label": "Fashion-MNIST",
    },
}


def build_fixture(dataset_key: str, per_class: int, num_classes: int = 10) -> Path:
    spec = DATASETS[dataset_key]

    # Test set lu sans transform : les PIL Images brutes suffisent.
    test_set = spec["loader"](root=str(ROOT_DIR / "data"), train=False, download=True)

    selected_images = []
    selected_labels = []
    remaining = {label: per_class for label in range(num_classes)}

    for image, label in test_set:
        label = int(label)
        if remaining.get(label, 0) <= 0:
            continue
        selected_images.append(np.array(image, dtype=np.uint8))
        selected_labels.append(label)
        remaining[label] -= 1
        if all(count <= 0 for count in remaining.values()):
            break

    missing = {label: count for label, count in remaining.items() if count > 0}
    if missing:
        raise RuntimeError(f"Images manquantes pour certaines classes : {missing}")

    # Tri par classe puis par ordre d'apparition : les index restent stables
    # entre deux generations, donc les liens partages vers la demo aussi.
    order = np.argsort(np.array(selected_labels), kind="stable")
    images = np.stack(selected_images)[order]
    labels = np.array(selected_labels, dtype=np.int64)[order]

    output_path = ASSETS_DIR / spec["filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images, labels=labels)

    size_kb = output_path.stat().st_size / 1024
    print(f"{spec['label']:15s} -> {output_path.name}")
    print(
        f"                   {images.shape[0]} images ({per_class} par classe), "
        f"{images.shape[1]}x{images.shape[2]}, {size_kb:.1f} Ko"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=12, help="Nombre d'images par classe (defaut : 12)")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS) + ["all"],
        default="all",
        help="Dataset a generer (defaut : all)",
    )
    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class doit etre >= 1")

    targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset_key in targets:
        build_fixture(dataset_key, args.per_class)


if __name__ == "__main__":
    main()
