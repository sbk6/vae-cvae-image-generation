"""Catalogue des datasets et des modeles servis par l'API.

Remplace l'ancien `registry.py`, qui ne connaissait qu'un seul dataset et une
seule famille de modeles.

Deux strategies de decouverte coexistent, pour de bonnes raisons :

- **MNIST** : liste explicite. Les checkpoints sont versionnes dans le depot,
  leurs chemins sont stables, et chacun merite une description redigee.
- **Fashion-MNIST** : decouverte par glob. Les checkpoints de David sont
  gitignores (`*.pt`) et arrivent hors du depot ; leurs noms de run dependent
  de ses arguments d'entrainement (`cvae_beta_01_seed42_final.pt`...). Figer
  une liste ici obligerait a la corriger a chaque fichier recu.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from backend.adapters import LOADERS, ModelAdapter
from src.training.trainer import get_device

ROOT_DIR = Path(__file__).resolve().parent.parent

FASHION_CHECKPOINT_DIR = ROOT_DIR / "projects" / "david_fashion_mnist" / "checkpoints"

MNIST_CLASS_NAMES = [str(index) for index in range(10)]
FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Pantalon",
    "Pull",
    "Robe",
    "Manteau",
    "Sandale",
    "Chemise",
    "Basket",
    "Sac",
    "Bottine",
]


@dataclass
class ModelEntry:
    """Description statique d'un modele servi par l'API."""

    model_id: str          # namespace par dataset : "mnist/cvae_main"
    dataset_id: str
    label: str
    description: str
    loader: str            # cle dans backend.adapters.LOADERS
    checkpoint_path: Path
    family: str            # "main" ou "ablation" : pilote l'onglet d'affichage
    beta: Optional[float] = None
    conditional: bool = False
    config_path: Optional[Path] = None
    # Cle de regroupement pour la comparaison a beta variable. Independante de
    # `family` : chez David les memes fichiers servent de modele principal ET
    # de serie d'ablation, alors que Sylvain a des checkpoints separes.
    ablation_series: Optional[str] = None
    adapter: Optional[ModelAdapter] = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists()


@dataclass
class DatasetEntry:
    """Un dataset et les modeles entraines dessus."""

    dataset_id: str
    label: str
    description: str
    class_names: List[str]
    fixture_name: str
    models: List[ModelEntry] = field(default_factory=list)

    @property
    def available_models(self) -> List[ModelEntry]:
        return [model for model in self.models if model.available]


# ------------------------------------------------------------------ MNIST

def _mnist_models() -> List[ModelEntry]:
    """Checkpoints MNIST, versionnes dans le depot donc listes explicitement."""
    experiments = ROOT_DIR / "reports" / "experiments"
    configs = ROOT_DIR / "configs"

    def entry(model_id, label, description, checkpoint, config, family, beta, conditional=False):
        return ModelEntry(
            model_id=f"mnist/{model_id}",
            dataset_id="mnist",
            label=label,
            description=description,
            loader="sylvain_mnist",
            checkpoint_path=checkpoint,
            config_path=config,
            family=family,
            beta=beta,
            conditional=conditional,
            # Seuls les checkpoints dedies a l'ablation forment la serie : les
            # modeles principaux ont ete entraines sur le dataset complet et ne
            # sont donc pas comparables aux runs d'ablation (sous-echantillonnes).
            ablation_series="vae" if family == "ablation" else None,
        )

    return [
        entry(
            "vae_main", "VAE",
            "VAE non conditionnel, 10 epochs sur MNIST complet, latent 16.",
            experiments / "vae_main" / "best_checkpoint.pth",
            configs / "mnist_vae.yaml", "main", 1.0,
        ),
        entry(
            "cvae_main", "CVAE",
            "VAE conditionnel (label injecte en canaux), memes reglages que le VAE.",
            experiments / "cvae_main" / "best_checkpoint.pth",
            configs / "mnist_cvae.yaml", "main", 1.0, conditional=True,
        ),
        entry(
            "ablation_beta_0.1", "β = 0.1",
            "Ablation : KL faiblement penalise, reconstruction fidele mais latent peu regulier.",
            experiments / "ablation" / "beta_0.1" / "best_checkpoint.pth",
            configs / "ablation_beta.yaml", "ablation", 0.1,
        ),
        entry(
            "ablation_beta_1.0", "β = 1.0",
            "Ablation : VAE standard, meilleur compromis reconstruction / regularite.",
            experiments / "ablation" / "beta_1.0" / "best_checkpoint.pth",
            configs / "ablation_beta.yaml", "ablation", 1.0,
        ),
        entry(
            "ablation_beta_5.0", "β = 5.0",
            "Ablation : KL fortement penalise, effondrement du posterior (KL ~ 0.56).",
            experiments / "ablation" / "beta_5.0" / "best_checkpoint.pth",
            configs / "ablation_beta.yaml", "ablation", 5.0,
        ),
    ]


# ---------------------------------------------------------- Fashion-MNIST

_BETA_TAG = re.compile(r"beta[_-]?([0-9]+)", re.IGNORECASE)


def parse_beta_tag(tag: str) -> Optional[float]:
    """Traduit le suffixe beta d'un nom de fichier en valeur numerique.

    Convention de David, lisible dans ses noms de resultats
    (`cvae_beta_01_history.csv`, `vae_beta_4_generations.png`) : le point
    decimal est supprime, donc "01" designe 0.1 et "4" designe 4.
    """
    if not tag:
        return None
    if tag.startswith("0") and len(tag) > 1:
        return float("0." + tag[1:])
    try:
        return float(tag)
    except ValueError:
        return None


def _fashion_models(checkpoint_dir: Path = FASHION_CHECKPOINT_DIR) -> List[ModelEntry]:
    """Decouvre les checkpoints Fashion-MNIST presents sur le disque.

    Les poids de David sont gitignores : ils sont deposes manuellement dans
    `projects/david_fashion_mnist/checkpoints/`. Tant que le dossier est
    absent, le dataset est simplement annonce comme indisponible plutot que de
    faire echouer le demarrage.
    """
    if not checkpoint_dir.is_dir():
        return []

    entries: List[ModelEntry] = []
    for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
        name = checkpoint.stem
        # "cvae" doit etre teste avant "vae", qui en est un suffixe.
        conditional = name.lower().startswith("cvae") or "_cvae" in name.lower()

        match = _BETA_TAG.search(name)
        beta = parse_beta_tag(match.group(1)) if match else None

        kind = "CVAE" if conditional else "VAE"
        label = f"{kind} β = {beta:g}" if beta is not None else f"{kind} ({name})"

        entries.append(
            ModelEntry(
                model_id=f"fashion/{name}",
                dataset_id="fashion_mnist",
                label=label,
                description=f"{kind} Fashion-MNIST entraine par David — checkpoint {checkpoint.name}.",
                loader="david_fashion",
                checkpoint_path=checkpoint,
                # beta = 1 est le run de reference affiche dans les onglets
                # principaux, mais il fait aussi partie de la serie d'ablation :
                # tous ces runs partagent les memes reglages hormis beta.
                family="main" if beta == 1.0 else "ablation",
                beta=beta,
                conditional=conditional,
                ablation_series="cvae" if conditional else "vae",
            )
        )
    return entries


# ---------------------------------------------------------------- catalogue

def build_datasets() -> Dict[str, DatasetEntry]:
    return {
        "mnist": DatasetEntry(
            dataset_id="mnist",
            label="MNIST",
            description="Chiffres manuscrits 28x28. Modeles de src/, sortie Tanh dans [-1, 1].",
            class_names=MNIST_CLASS_NAMES,
            fixture_name="mnist_samples.npz",
            models=_mnist_models(),
        ),
        "fashion_mnist": DatasetEntry(
            dataset_id="fashion_mnist",
            label="Fashion-MNIST",
            description="Vetements 28x28. Modeles de projects/david_fashion_mnist/, sortie Sigmoid dans [0, 1].",
            class_names=FASHION_CLASS_NAMES,
            fixture_name="fashion_mnist_samples.npz",
            models=_fashion_models(),
        ),
    }


class Catalog:
    """Charge et met en cache les adaptateurs. Thread-safe (Flask sert en multi-thread)."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device: torch.device = get_device(device or "auto")
        self.datasets = build_datasets()
        self._models: Dict[str, ModelEntry] = {
            model.model_id: model
            for dataset in self.datasets.values()
            for model in dataset.models
        }
        self._lock = threading.Lock()

    # -------------------------------------------------------------- lecture

    def available_datasets(self) -> List[DatasetEntry]:
        """Datasets ayant au moins un checkpoint utilisable."""
        return [dataset for dataset in self.datasets.values() if dataset.available_models]

    def dataset(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self.datasets:
            raise KeyError(f"Dataset inconnu : {dataset_id}")
        return self.datasets[dataset_id]

    def models(self, dataset_id: Optional[str] = None) -> List[ModelEntry]:
        if dataset_id is None:
            return [model for model in self._models.values() if model.available]
        return self.dataset(dataset_id).available_models

    # ------------------------------------------------------------ chargement

    def adapter(self, model_id: str) -> ModelAdapter:
        """Renvoie l'adaptateur du modele, en le chargeant au premier appel."""
        entry = self._models.get(model_id)
        if entry is None:
            raise KeyError(f"Modele inconnu : {model_id}")

        if entry.adapter is not None:
            return entry.adapter

        with self._lock:
            # Un autre thread a pu charger le modele pendant l'attente du verrou.
            if entry.adapter is not None:
                return entry.adapter

            if not entry.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint absent pour '{model_id}' : {entry.checkpoint_path}"
                )

            loader = LOADERS[entry.loader]
            entry.adapter = loader(
                entry.checkpoint_path,
                self.device,
                config_path=entry.config_path,
            )
            return entry.adapter

    def entry(self, model_id: str) -> ModelEntry:
        if model_id not in self._models:
            raise KeyError(f"Modele inconnu : {model_id}")
        return self._models[model_id]

    # ----------------------------------------------------------- serialisation

    def dataset_metadata(self, dataset: DatasetEntry) -> dict:
        return {
            "id": dataset.dataset_id,
            "label": dataset.label,
            "description": dataset.description,
            "class_names": dataset.class_names,
            "model_count": len(dataset.available_models),
        }

    def model_metadata(self, entry: ModelEntry) -> dict:
        adapter = entry.adapter
        return {
            "id": entry.model_id,
            "dataset": entry.dataset_id,
            "label": entry.label,
            "description": entry.description,
            "family": entry.family,
            "beta": entry.beta,
            "conditional": entry.conditional,
            "latent_dim": adapter.latent_dim if adapter else None,
            "num_conditions": adapter.num_conditions if adapter else None,
            "loaded": adapter is not None,
        }
