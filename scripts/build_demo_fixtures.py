"""Precalcule des echantillons d'images reelles pour la demo web.

Pourquoi : la demo a besoin de vraies images (reconstruction, interpolation),
mais embarquer torchvision + les datasets complets dans l'image Docker
couterait ~64 Mo par dataset et un telechargement au premier demarrage. On
extrait donc une fois pour toutes quelques images par classe dans des .npz de
~20 Ko, versionnes avec le code.

Les images sont stockees en uint8 [0, 255], **sans normalisation** : chaque
famille de modeles attend sa propre plage ([-1, 1] pour MNIST et CelebA,
[0, 1] pour Fashion-MNIST). La normalisation est appliquee cote API par
l'adaptateur du modele cible.

CelebA est un cas a part : contrairement a MNIST/Fashion-MNIST (telecharges
a la volee via torchvision), ses images ne sont pas rechargees ici. Le
fixture est construit a partir du cache local deja produit par
projects/blaise_celeba/ (data_cache/test_*.npz), pour ne pas faire dependre
ce script partage de la librairie `datasets` (Hugging Face), qui n'est
installee que dans l'environnement virtuel du sous-projet CelebA. Il faut
donc avoir lance au moins un entrainement ou une evaluation CelebA au
prealable pour que ce cache existe.

Usage :
    python scripts/build_demo_fixtures.py                  # tous les datasets
    python scripts/build_demo_fixtures.py --dataset mnist
    python scripts/build_demo_fixtures.py --dataset celeba
    python scripts/build_demo_fixtures.py --per-class 16
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from torchvision import datasets

ASSETS_DIR = ROOT_DIR / "backend" / "assets"
CELEBA_CACHE_DIR = ROOT_DIR / "projects" / "blaise_celeba" / "data_cache"

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


def _find_celeba_test_cache() -> Path:
    """Le nom exact du fichier de cache depend de n_test/seed/attributs (voir
    projects/blaise_celeba/data/dataset.py, _cache_path) : on prend le plus
    recent plutot que de deviner le nom complet ici."""
    candidates = sorted(CELEBA_CACHE_DIR.glob("test_*.npz"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(
            "Aucun cache CelebA trouve dans projects/blaise_celeba/data_cache/. "
            "Lancer d'abord un entrainement ou une evaluation CelebA, par exemple : "
            "cd projects/blaise_celeba && .venv/bin/python -m training.train --config configs/celeba_vae.yaml"
        )
    return candidates[-1]


def build_celeba_fixture(per_combo: int) -> Path:
    """Construit le fixture CelebA a partir du cache local (voir docstring du module)."""
    cache_path = _find_celeba_test_cache()
    with np.load(cache_path) as data:
        images = data["images"]          # (N, H, W, 3) uint8
        attributes = data["attributes"]  # (N, num_attrs) float32 dans {0, 1}

    num_attrs = attributes.shape[1]
    combos = list(itertools.product([0, 1], repeat=num_attrs))

    selected_images, selected_labels = [], []
    for combo_index, combo in enumerate(combos):
        matches = np.where((attributes == np.array(combo, dtype=np.float32)).all(axis=1))[0]
        for match_index in matches[:per_combo]:
            selected_images.append(images[match_index])
            selected_labels.append(combo_index)

    counts = np.bincount(selected_labels, minlength=len(combos))
    under_represented = [i for i, count in enumerate(counts) if count < per_combo]
    if under_represented:
        print(
            f"Attention : moins de {per_combo} exemples de test pour les combinaisons "
            f"{[combos[i] for i in under_represented]} (cache trop petit ou attribut rare)."
        )

    images_arr = np.stack(selected_images)
    labels_arr = np.array(selected_labels, dtype=np.int64)

    output_path = ASSETS_DIR / "celeba_samples.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images_arr, labels=labels_arr)

    size_kb = output_path.stat().st_size / 1024
    print(f"{'CelebA':15s} -> {output_path.name}")
    print(
        f"                   {images_arr.shape[0]} images ({per_combo} par combinaison d'attributs), "
        f"{images_arr.shape[1]}x{images_arr.shape[2]}x3, {size_kb:.1f} Ko"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=12, help="Nombre d'images par classe/combinaison (defaut : 12)")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS) + ["celeba", "all"],
        default="all",
        help="Dataset a generer (defaut : all)",
    )
    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class doit etre >= 1")

    targets = sorted(list(DATASETS) + ["celeba"]) if args.dataset == "all" else [args.dataset]
    for dataset_key in targets:
        if dataset_key == "celeba":
            build_celeba_fixture(args.per_class)
        else:
            build_fixture(dataset_key, args.per_class)


if __name__ == "__main__":
    main()
