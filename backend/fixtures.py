"""Acces aux images reelles utilisees par la demo (MNIST et Fashion-MNIST).

La reconstruction et l'interpolation ont besoin de vraies images en entree.
Plutot que d'embarquer torchvision + les datasets complets (~64 Mo chacun)
dans l'image Docker, on precalcule un petit echantillon par dataset avec
`scripts/build_demo_fixtures.py`. Chaque fichier pese ~20 Ko.

Les images sont stockees en uint8 [0, 255], sans normalisation : c'est
l'adaptateur du modele cible qui applique la sienne, puisque les deux familles
de modeles n'attendent pas la meme plage ([-1, 1] contre [0, 1]).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "backend" / "assets"


class FixtureStore:
    """Charge un fixture d'images et le sert par index ou par classe."""

    def __init__(self, filename: str, assets_dir: Path = ASSETS_DIR) -> None:
        self.path = assets_dir / filename
        self._images: Optional[np.ndarray] = None  # (N, H, W) uint8
        self._labels: Optional[np.ndarray] = None  # (N,) int64

    @property
    def available(self) -> bool:
        return self.path.exists()

    def _ensure_loaded(self) -> None:
        if self._images is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(
                f"Fixture d'images absent : {self.path}. "
                "Le generer avec : python scripts/build_demo_fixtures.py"
            )
        with np.load(self.path) as data:
            self._images = data["images"]
            self._labels = data["labels"]

    def __len__(self) -> int:
        if not self.available:
            return 0
        self._ensure_loaded()
        return int(self._images.shape[0])

    def indices_by_class(self) -> Dict[int, List[int]]:
        """Index des images disponibles, regroupes par classe."""
        self._ensure_loaded()
        grouped: Dict[int, List[int]] = {}
        for index, label in enumerate(self._labels.tolist()):
            grouped.setdefault(int(label), []).append(index)
        return grouped

    def label_of(self, index: int) -> int:
        self._ensure_loaded()
        self._check_index(index)
        return int(self._labels[index])

    def _check_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("L'index doit etre un entier.")
        if index < 0 or index >= self._images.shape[0]:
            raise ValueError(f"Index hors bornes : {index} (0..{self._images.shape[0] - 1}).")

    def raw(self, index: int) -> np.ndarray:
        """Image brute (H, W) en uint8, a normaliser par l'adaptateur cible."""
        self._ensure_loaded()
        self._check_index(index)
        return self._images[index]

    def first_index_of_class(self, class_label: int) -> int:
        grouped = self.indices_by_class()
        if class_label not in grouped:
            raise ValueError(f"Aucune image disponible pour la classe {class_label}.")
        return grouped[class_label][0]


class FixtureRegistry:
    """Un FixtureStore par dataset, cree a la demande."""

    def __init__(self) -> None:
        self._stores: Dict[str, FixtureStore] = {}

    def get(self, dataset_id: str, filename: str) -> FixtureStore:
        if dataset_id not in self._stores:
            self._stores[dataset_id] = FixtureStore(filename)
        return self._stores[dataset_id]
