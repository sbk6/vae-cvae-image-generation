"""Chargement d'un sous-echantillon de CelebA avec attributs, pour le VAE/CVAE.

CelebA officiel (mmlab.ie.cuhk.edu.hk/projects/CelebA.html) se telecharge par
defaut via `torchvision.datasets.CelebA`, qui pointe vers un lien Google
Drive partage. Ce lien renvoie tres frequemment une erreur de quota depasse
("Google Drive - Quota exceeded") au lieu du fichier attendu : probleme
recurrent et documente (issues GitHub pytorch/vision #1920, forum PyTorch),
qui ne depend pas de la machine utilisee.

Solution retenue ici : le miroir Hugging Face `tpremoli/CelebA-attrs`, qui
republie les memes images avec les memes 40 attributs binaires et les memes
splits officiels train/validation/test, au format parquet, sans passer par
Google Drive. C'est un choix documente et assume, pas un contournement
silencieux.

Le sujet demande explicitement un sous-echantillon de CelebA : on ne charge
donc jamais l'integralite des ~203 000 images. Le sous-echantillon est tire
aleatoirement (seed fixe) apres melange du split complet, plutot que de
prendre les N premieres images : CelebA est ordonne par identite (plusieurs
photos consecutives de la meme personne), donc prendre un prefixe biaiserait
l'echantillon vers un petit nombre de visages.

Une fois tire, le sous-echantillon est mis en cache localement (images deja
redimensionnees, deja converties en tableau numpy) pour ne pas repayer le
telechargement ni le redimensionnement a chaque lancement.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HF_DATASET_NAME = "tpremoli/CelebA-attrs"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

# Attributs de conditionnement retenus pour le CVAE. Choisis pour deux
# raisons, verifiees empiriquement sur le sous-echantillon avant de figer
# cette liste (voir evaluation/inspect_attributes.py) :
# 1. distinctifs visuellement, donc verifiables a l'oeil sur les figures ;
# 2. suffisamment equilibres (aucun trop rare) pour que les 2**3 combinaisons
#    possibles du vecteur multi-hot aient chacune assez d'exemples reels en
#    entrainement, et evitent qu'une combinaison soit quasiment vide.
DEFAULT_ATTRIBUTES: List[str] = ["Smiling", "Male", "Wavy_Hair"]

HF_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


@dataclass
class DatasetInfo:
    image_size: Tuple[int, int]
    channels: int
    num_conditions: int
    attribute_names: List[str]


class CelebASubsample(Dataset):
    """Sous-echantillon CelebA deja charge en memoire (images + attributs)."""

    def __init__(self, images: np.ndarray, attributes: np.ndarray) -> None:
        # images : (N, H, W, 3) uint8, RGB.
        # attributes : (N, num_conditions) float32, valeurs dans {0.0, 1.0}.
        assert images.shape[0] == attributes.shape[0]
        self.images = images
        self.attributes = attributes

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        # [0, 255] uint8 -> [-1, 1] float32, cf. models/vae.py (decodeur Tanh).
        image = self.images[index].astype(np.float32) / 255.0
        image = image * 2.0 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        attrs = torch.from_numpy(self.attributes[index])
        return image, attrs


def _cache_path(split: str, n_samples: int, seed: int, image_size: Tuple[int, int], attributes: Sequence[str]) -> Path:
    attrs_tag = "-".join(attributes)
    name = f"{split}_n{n_samples}_seed{seed}_{image_size[0]}x{image_size[1]}_{attrs_tag}.npz"
    return CACHE_DIR / name


def _download_split(split: str, n_samples: int, seed: int, image_size: Tuple[int, int], attributes: Sequence[str]):
    """Telecharge le split HF, tire un sous-echantillon reproductible, redimensionne."""
    from datasets import load_dataset
    from PIL import Image

    hf_split = HF_SPLIT_NAMES[split]
    print(f"[data] Telechargement/chargement du split '{hf_split}' depuis {HF_DATASET_NAME} ...")
    hf_dataset = load_dataset(HF_DATASET_NAME, split=hf_split)

    available = set(hf_dataset.column_names)
    missing = [name for name in attributes if name not in available]
    if missing:
        raise ValueError(
            f"Attribut(s) introuvable(s) dans {HF_DATASET_NAME} : {missing}. "
            f"Colonnes disponibles (hors 'image') : {sorted(a for a in available if a != 'image')}"
        )

    n_available = len(hf_dataset)
    n_take = min(n_samples, n_available)
    # Melange avec seed fixe puis prefixe : reproductible, et evite le biais
    # d'ordre par identite explique dans le docstring du module.
    shuffled = hf_dataset.shuffle(seed=seed).select(range(n_take))

    images = np.zeros((n_take, image_size[0], image_size[1], 3), dtype=np.uint8)
    attrs = np.zeros((n_take, len(attributes)), dtype=np.float32)

    for index, row in enumerate(shuffled):
        image = row["image"].convert("RGB").resize(image_size, Image.BILINEAR)
        images[index] = np.array(image, dtype=np.uint8)
        # Les attributs du miroir HF sont encodes en -1/1 (convention CelebA
        # officielle) ; on les ramene en 0/1 pour le vecteur multi-hot du CVAE.
        for attr_index, attr_name in enumerate(attributes):
            attrs[index, attr_index] = 1.0 if row[attr_name] > 0 else 0.0

    return images, attrs


def load_split(
    split: str,
    n_samples: int,
    seed: int = 42,
    image_size: Tuple[int, int] = (64, 64),
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Renvoie (images, attributs) pour un split, en utilisant le cache local si present."""
    attributes = list(attributes)
    cache_path = _cache_path(split, n_samples, seed, image_size, attributes)
    if cache_path.exists():
        with np.load(cache_path) as data:
            return data["images"], data["attributes"]

    images, attrs = _download_split(split, n_samples, seed, image_size, attributes)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, images=images, attributes=attrs)
    print(f"[data] Sous-echantillon '{split}' mis en cache : {cache_path} ({images.shape[0]} images)")
    return images, attrs


def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Construit les DataLoaders train/val/test a partir d'un dict de configuration.

    Signature volontairement proche de `src.data.datasets.build_dataloaders`
    (Sylvain) pour que les scripts d'entrainement des deux sous-projets se
    lisent de la meme facon, meme si l'implementation est independante.
    """
    dataset_cfg = config["dataset"]
    image_size = tuple(dataset_cfg.get("image_size", [64, 64]))
    attributes = dataset_cfg.get("attributes", DEFAULT_ATTRIBUTES)
    seed = dataset_cfg.get("seed", 42)

    n_train = dataset_cfg.get("n_train", 8000)
    n_val = dataset_cfg.get("n_val", 1500)
    n_test = dataset_cfg.get("n_test", 1500)

    train_images, train_attrs = load_split("train", n_train, seed, image_size, attributes)
    val_images, val_attrs = load_split("val", n_val, seed, image_size, attributes)
    test_images, test_attrs = load_split("test", n_test, seed, image_size, attributes)

    train_set = CelebASubsample(train_images, train_attrs)
    val_set = CelebASubsample(val_images, val_attrs)
    test_set = CelebASubsample(test_images, test_attrs)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    dataset_info = DatasetInfo(
        image_size=image_size,
        channels=3,
        num_conditions=len(attributes),
        attribute_names=list(attributes),
    )
    return train_loader, val_loader, test_loader, dataset_info
