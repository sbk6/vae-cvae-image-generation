"""Verifie la frequence des attributs candidats sur un sous-echantillon CelebA.

Sert a choisir les attributs de conditionnement du CVAE (data/dataset.py,
DEFAULT_ATTRIBUTES) en connaissance de cause : un attribut trop rare
(par exemple "Eyeglasses", present sur environ 6% des visages dans CelebA
complet) donnerait des combinaisons quasi vides dans le vecteur multi-hot du
CVAE, avec trop peu d'exemples reels pour bien apprendre ces combinaisons.

Usage :
    python -m evaluation.inspect_attributes --n-samples 4000
    python -m evaluation.inspect_attributes --config configs/celeba_vae.yaml --split train
"""
import argparse
from collections import Counter

from datasets import load_dataset

from data.dataset import DEFAULT_ATTRIBUTES, build_dataloaders
from utils import load_yaml_config

CANDIDATE_ATTRIBUTES = [
    "Smiling", "Male", "Wavy_Hair", "Eyeglasses", "Blond_Hair", "Young",
    "Heavy_Makeup", "Wearing_Hat", "Bald", "Mustache", "Black_Hair", "Brown_Hair",
]


def print_candidate_frequencies(rows, n: int, title: str) -> None:
    counters = {name: Counter() for name in CANDIDATE_ATTRIBUTES}
    for row in rows:
        for name in CANDIDATE_ATTRIBUTES:
            counters[name][row[name] > 0] += 1

    print(f"\n{title}\n")
    print(f"{'attribut':22s} {'positif':>8s} {'%':>7s}")
    for name in CANDIDATE_ATTRIBUTES:
        positive = counters[name][True]
        pct = 100.0 * positive / n
        print(f"{name:22s} {positive:8d} {pct:6.1f}%")


def print_combo_counts_from_rows(rows, n: int, attributes=DEFAULT_ATTRIBUTES) -> None:
    print(f"\nCombinaisons pour les attributs ({', '.join(attributes)}) :")
    combo_counts = Counter()
    for row in rows:
        key = tuple(row[name] > 0 for name in attributes)
        combo_counts[key] += 1
    print_combo_counts(combo_counts, n, attributes)


def print_combo_counts(combo_counts: Counter, n: int, attributes=DEFAULT_ATTRIBUTES) -> None:
    for key in sorted(combo_counts):
        label = " ".join(f"{name}={int(value)}" for name, value in zip(attributes, key))
        print(f"  {label:40s} {combo_counts[key]:6d}  ({100.0 * combo_counts[key] / n:5.1f}%)")


def inspect_from_config(config_path: str, split: str) -> None:
    config = load_yaml_config(config_path)
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[split]

    combo_counts = Counter()
    n = 0
    for _, attrs in loader:
        for row in attrs:
            key = tuple(bool(value.item() > 0.5) for value in row)
            combo_counts[key] += 1
            n += 1

    dataset_cfg = config["dataset"]
    print(
        f"\nDistribution reellement utilisee par le DataLoader '{split}' "
        f"({n} images, strategy={dataset_cfg.get('sampling_strategy', 'random')}) :"
    )
    print_combo_counts(combo_counts, n, dataset_info.attribute_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, help="Inspecte le DataLoader construit depuis une config YAML")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    if args.config:
        inspect_from_config(args.config, args.split)
        return

    print(f"[inspect] Chargement du split 'train' (tpremoli/CelebA-attrs) ...")
    dataset = load_dataset("tpremoli/CelebA-attrs", split="train")
    shuffled = dataset.shuffle(seed=args.seed).select(range(min(args.n_samples, len(dataset))))

    n = len(shuffled)
    print_candidate_frequencies(
        shuffled,
        n,
        f"Frequence positive naturelle sur {n} images echantillonnees aleatoirement (seed={args.seed}) :",
    )
    print_combo_counts_from_rows(shuffled, n)


if __name__ == "__main__":
    main()
