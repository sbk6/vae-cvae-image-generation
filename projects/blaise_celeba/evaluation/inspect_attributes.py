"""Verifie la frequence des attributs candidats sur un sous-echantillon CelebA.

Sert a choisir les attributs de conditionnement du CVAE (data/dataset.py,
DEFAULT_ATTRIBUTES) en connaissance de cause : un attribut trop rare
(par exemple "Eyeglasses", present sur environ 6% des visages dans CelebA
complet) donnerait des combinaisons quasi vides dans le vecteur multi-hot du
CVAE, avec trop peu d'exemples reels pour bien apprendre ces combinaisons.

Usage :
    python -m evaluation.inspect_attributes --n-samples 4000
"""
import argparse
from collections import Counter

from datasets import load_dataset

CANDIDATE_ATTRIBUTES = [
    "Smiling", "Male", "Wavy_Hair", "Eyeglasses", "Blond_Hair", "Young",
    "Heavy_Makeup", "Wearing_Hat", "Bald", "Mustache", "Black_Hair", "Brown_Hair",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[inspect] Chargement du split 'train' (tpremoli/CelebA-attrs) ...")
    dataset = load_dataset("tpremoli/CelebA-attrs", split="train")
    shuffled = dataset.shuffle(seed=args.seed).select(range(min(args.n_samples, len(dataset))))

    counters = {name: Counter() for name in CANDIDATE_ATTRIBUTES}
    for row in shuffled:
        for name in CANDIDATE_ATTRIBUTES:
            counters[name][row[name] > 0] += 1

    n = len(shuffled)
    print(f"\nFrequence positive sur {n} images echantillonnees (seed={args.seed}) :\n")
    print(f"{'attribut':22s} {'positif':>8s} {'%':>7s}")
    for name in CANDIDATE_ATTRIBUTES:
        positive = counters[name][True]
        pct = 100.0 * positive / n
        print(f"{name:22s} {positive:8d} {pct:6.1f}%")

    print("\nCombinaisons pour les 3 attributs retenus par defaut (Smiling, Male, Wavy_Hair) :")
    combo_counts = Counter()
    for row in shuffled:
        key = (row["Smiling"] > 0, row["Male"] > 0, row["Wavy_Hair"] > 0)
        combo_counts[key] += 1
    for key in sorted(combo_counts):
        smiling, male, wavy = key
        label = f"Smiling={int(smiling)} Male={int(male)} Wavy_Hair={int(wavy)}"
        print(f"  {label:40s} {combo_counts[key]:6d}  ({100.0 * combo_counts[key] / n:5.1f}%)")


if __name__ == "__main__":
    main()
