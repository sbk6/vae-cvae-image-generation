"""
Évaluation multi-seed contrôlée de la diversité des générations CVAE.

Objectif
--------
Évaluer la robustesse de la diversité générative du CVAE final beta=1
sur plusieurs seeds d'entraînement, tout en contrôlant les vecteurs latents.

Protocole final
---------------
Par défaut, le script compare trois checkpoints CVAE beta=1 :

- training_seed = 0 ;
- training_seed = 42 ;
- training_seed = 123.

Le split train/validation du protocole reste fixé à :

    split_seed = 42

Une seule banque de vecteurs latents z ~ N(0, I) est créée sur CPU.
La même banque est réutilisée :

- pour les trois seeds d'entraînement ;
- pour les dix classes Fashion-MNIST.

Ainsi, les différences observées ne proviennent pas de tirages latents
différents.

Métriques
---------
Deux familles de métriques sont calculées, classe par classe puis
agrégées en moyenne macro sur les dix classes :

1. Diversité pixel
   RMS pixel global entre les paires de générations d'une même classe.

2. Diversité sémantique
   Distance cosinus moyenne entre les représentations pénultièmes du
   classifieur Fashion-MNIST indépendant.

Chaque métrique est calculée :

- sur toutes les images générées ;
- uniquement sur les générations jugées cohérentes avec la classe
  demandée par le classifieur indépendant.

La seconde version est particulièrement importante : elle évite de
récompenser artificiellement un modèle qui serait "divers" surtout
parce qu'il produit des images hors classe.

Sorties
-------
Le script crée notamment :

- generation_diversity_multiseed_runs.csv
- generation_diversity_multiseed_summary.csv
- generation_diversity_multiseed_per_class.csv
- generation_diversity_multiseed_per_class_summary.csv
- generation_diversity_pixel_coherent_by_seed.png
- generation_diversity_feature_coherent_by_seed.png
- generation_diversity_pixel_per_class_mean_std.png
- generation_diversity_feature_per_class_mean_std.png

Le jeu officiel de test Fashion-MNIST n'est jamais utilisé.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.cvae import CVAE
from models.fashion_classifier import FashionMNISTClassifier


# ============================================================
# CONSTANTES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DEFAULT_CVAE_CHECKPOINTS = [
    PROJECT_ROOT
    / "checkpoints"
    / "cvae_beta_1_seed0_gpu_multiseed.pt",

    PROJECT_ROOT
    / "checkpoints"
    / "cvae_beta_1_seed42_final.pt",

    PROJECT_ROOT
    / "checkpoints"
    / "cvae_beta_1_seed123_gpu_multiseed.pt",
]

DEFAULT_CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "fashion_classifier_seed42_final.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation_diversity_multiseed_final"
)

DEFAULT_MLFLOW_EXPERIMENT = (
    "fashion_mnist_final_evaluation"
)

DEFAULT_MLFLOW_RUN_NAME = (
    "cvae_beta1_multiseed_generation_diversity"
)


# ============================================================
# STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class LoadedCVAE:
    """
    Informations utiles associées à un checkpoint CVAE.
    """

    model: CVAE
    checkpoint_path: Path
    beta: float
    best_epoch: int | None
    latent_dim: int
    hidden_dim: int
    num_classes: int
    training_seed: int
    checkpoint_split_seed: int | None


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Lire et valider les arguments de la ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Mesurer la diversité multi-seed du CVAE final "
            "beta=1 sur Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=DEFAULT_CVAE_CHECKPOINTS,
        help=(
            "Checkpoints CVAE finaux à comparer. "
            "Par défaut : beta=1, seeds 0, 42 et 123."
        ),
    )

    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
        help=(
            "Checkpoint du classifieur Fashion-MNIST indépendant."
        ),
    )

    parser.add_argument(
        "--expected-beta",
        type=float,
        default=1.0,
        help=(
            "Valeur de beta attendue dans tous les checkpoints. "
            "Valeur par défaut : 1."
        ),
    )

    parser.add_argument(
        "--expected-training-seeds",
        nargs="+",
        type=int,
        default=[0, 42, 123],
        help=(
            "Seeds d'entraînement attendues. "
            "Par défaut : 0 42 123."
        ),
    )

    parser.add_argument(
        "--split-seed-protocol",
        type=int,
        default=42,
        help=(
            "Seed du split train/validation du protocole final. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help=(
            "Nombre d'images générées par classe et par seed. "
            "Valeur par défaut : 1000."
        ),
    )

    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=256,
        help=(
            "Taille des lots de génération/classification. "
            "Valeur par défaut : 256."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed de la banque latente partagée. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Appareil de calcul. "
            "Valeur par défaut : auto."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Dossier de sauvegarde des résultats."
        ),
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help=(
            "Nom de l'expérience MLflow."
        ),
    )

    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=DEFAULT_MLFLOW_RUN_NAME,
        help=(
            "Nom du run MLflow."
        ),
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "URI MLflow optionnelle. "
            "Sans valeur, SQLite local est utilisé."
        ),
    )

    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help=(
            "Désactive MLflow pour un test technique."
        ),
    )

    args = parser.parse_args()

    if len(args.checkpoints) == 0:
        parser.error(
            "Au moins un checkpoint CVAE doit être fourni."
        )

    if len(args.expected_training_seeds) == 0:
        parser.error(
            "--expected-training-seeds ne peut pas être vide."
        )

    if (
        len(set(args.expected_training_seeds))
        != len(args.expected_training_seeds)
    ):
        parser.error(
            "--expected-training-seeds contient des doublons."
        )

    if args.samples_per_class < 2:
        parser.error(
            "--samples-per-class doit être >= 2."
        )

    if args.generation_batch_size < 1:
        parser.error(
            "--generation-batch-size doit être >= 1."
        )

    if not args.mlflow_experiment_name.strip():
        parser.error(
            "--mlflow-experiment-name ne peut pas être vide."
        )

    if not args.mlflow_run_name.strip():
        parser.error(
            "--mlflow-run-name ne peut pas être vide."
        )

    return args


# ============================================================
# CHEMINS
# ============================================================

def resolve_project_path(
    path: Path,
) -> Path:
    """
    Résoudre un chemin relatif depuis la racine du sous-projet.
    """

    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


# ============================================================
# APPAREIL ET REPRODUCTIBILITÉ
# ============================================================

def select_device(
    requested: str,
) -> torch.device:
    """
    Choisir CPU ou CUDA.
    """

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA a été demandé mais aucun GPU CUDA "
                "n'est disponible."
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(
    seed: int,
) -> None:
    """
    Fixer les principales sources d'aléa.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHARGEMENT GÉNÉRIQUE DES CHECKPOINTS
# ============================================================

def load_torch_checkpoint(
    path: Path,
    device: torch.device,
) -> Any:
    """
    Charger un checkpoint PyTorch.
    """

    path = resolve_project_path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}"
        )

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device,
        )


def nested_checkpoint_value(
    checkpoint: Any,
    names: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Rechercher une valeur dans plusieurs zones usuelles
    d'un checkpoint.

    Le champ ``configuration`` est explicitement inclus car
    c'est celui utilisé par les scripts d'entraînement du projet.
    """

    if not isinstance(checkpoint, dict):
        return default

    scopes = [
        checkpoint
    ]

    for key in (
        "configuration",
        "config",
        "model_config",
        "hyperparameters",
        "hparams",
        "params",
        "training_config",
    ):

        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            scopes.append(
                value
            )

    for scope in scopes:

        for name in names:

            if name in scope:
                return scope[
                    name
                ]

    return default


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    Extraire le state_dict depuis différents formats possibles.
    """

    if isinstance(
        checkpoint,
        nn.Module,
    ):
        return checkpoint.state_dict()

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Format de checkpoint non pris en charge : "
            f"{type(checkpoint).__name__}"
        )

    possible_keys = (
        "model_state_dict",
        "state_dict",
        "model",
        "network_state_dict",
    )

    for key in possible_keys:

        candidate = checkpoint.get(
            key
        )

        if (
            isinstance(
                candidate,
                dict,
            )
            and candidate
            and all(
                torch.is_tensor(
                    value
                )
                for value in candidate.values()
            )
        ):
            return candidate

    if (
        checkpoint
        and all(
            torch.is_tensor(
                value
            )
            for value in checkpoint.values()
        )
    ):
        return checkpoint

    raise KeyError(
        "Impossible de trouver le state_dict "
        "dans le checkpoint."
    )


def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Retirer le préfixe ``module.`` de DataParallel si nécessaire.
    """

    if (
        state_dict
        and all(
            key.startswith(
                "module."
            )
            for key in state_dict
        )
    ):

        return {
            key[
                len("module.") :
            ]: value
            for key, value in state_dict.items()
        }

    return state_dict


# ============================================================
# CHARGEMENT DU CVAE
# ============================================================

def instantiate_cvae(
    latent_dim: int,
    hidden_dim: int,
    num_classes: int,
) -> CVAE:
    """
    Construire un CVAE en respectant sa signature réelle.
    """

    signature = inspect.signature(
        CVAE.__init__
    )

    parameters = (
        signature.parameters
    )

    kwargs: dict[
        str,
        Any
    ] = {}

    if "latent_dim" in parameters:
        kwargs[
            "latent_dim"
        ] = latent_dim

    if "hidden_dim" in parameters:
        kwargs[
            "hidden_dim"
        ] = hidden_dim

    if "num_classes" in parameters:
        kwargs[
            "num_classes"
        ] = num_classes

    return CVAE(
        **kwargs
    )


def load_cvae_checkpoint(
    path: Path,
    device: torch.device,
) -> LoadedCVAE:
    """
    Charger un checkpoint CVAE avec ses métadonnées.
    """

    checkpoint_path = resolve_project_path(
        path
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        device,
    )

    model_type = nested_checkpoint_value(
        checkpoint,
        [
            "model_type",
        ],
        None,
    )

    if model_type != "CVAE":
        raise ValueError(
            "Le checkpoint ne correspond pas à un CVAE : "
            f"{checkpoint_path.name}. "
            f"model_type={model_type!r}"
        )

    latent_dim = int(
        nested_checkpoint_value(
            checkpoint,
            [
                "latent_dim",
            ],
            16,
        )
    )

    hidden_dim = int(
        nested_checkpoint_value(
            checkpoint,
            [
                "hidden_dim",
            ],
            256,
        )
    )

    num_classes = int(
        nested_checkpoint_value(
            checkpoint,
            [
                "num_classes",
                "n_classes",
            ],
            10,
        )
    )

    beta = float(
        nested_checkpoint_value(
            checkpoint,
            [
                "beta",
            ],
            float("nan"),
        )
    )

    best_epoch_value = (
        nested_checkpoint_value(
            checkpoint,
            [
                "best_epoch",
                "epoch",
            ],
            None,
        )
    )

    best_epoch = (
        None
        if best_epoch_value is None
        else int(
            best_epoch_value
        )
    )

    # Les nouveaux checkpoints contiennent training_seed.
    # Le checkpoint historique seed=42 contient seulement seed.
    training_seed_value = (
        nested_checkpoint_value(
            checkpoint,
            [
                "training_seed",
                "seed",
            ],
            None,
        )
    )

    if training_seed_value is None:
        raise KeyError(
            "Impossible d'identifier la seed d'entraînement "
            f"dans {checkpoint_path.name}."
        )

    training_seed = int(
        training_seed_value
    )

    checkpoint_split_seed_value = (
        nested_checkpoint_value(
            checkpoint,
            [
                "split_seed",
            ],
            None,
        )
    )

    checkpoint_split_seed = (
        None
        if checkpoint_split_seed_value is None
        else int(
            checkpoint_split_seed_value
        )
    )

    model = instantiate_cvae(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    state_dict = strip_module_prefix(
        extract_state_dict(
            checkpoint
        )
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return LoadedCVAE(
        model=model,
        checkpoint_path=(
            checkpoint_path
        ),
        beta=beta,
        best_epoch=best_epoch,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        training_seed=(
            training_seed
        ),
        checkpoint_split_seed=(
            checkpoint_split_seed
        ),
    )


def validate_loaded_models(
    loaded_models: Sequence[LoadedCVAE],
    expected_beta: float,
    expected_training_seeds: Sequence[int],
    split_seed_protocol: int,
) -> list[LoadedCVAE]:
    """
    Auditer les checkpoints et les trier selon l'ordre des seeds attendu.
    """

    if not loaded_models:
        raise RuntimeError(
            "Aucun CVAE chargé."
        )

    expected_seeds = list(
        expected_training_seeds
    )

    if len(loaded_models) != len(
        expected_seeds
    ):
        raise ValueError(
            "Le nombre de checkpoints ne correspond pas "
            "au nombre de seeds attendues. "
            f"Checkpoints={len(loaded_models)}, "
            f"seeds={len(expected_seeds)}."
        )

    observed_seeds = [
        loaded.training_seed
        for loaded in loaded_models
    ]

    if (
        len(set(observed_seeds))
        != len(observed_seeds)
    ):
        raise ValueError(
            "Plusieurs checkpoints possèdent la même "
            f"training_seed : {observed_seeds}."
        )

    if set(observed_seeds) != set(
        expected_seeds
    ):
        raise ValueError(
            "Les training_seeds observées ne correspondent pas "
            "au protocole attendu. "
            f"Observées={sorted(observed_seeds)}, "
            f"attendues={sorted(expected_seeds)}."
        )

    for loaded in loaded_models:

        if not math.isclose(
            loaded.beta,
            expected_beta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Beta inattendu dans {loaded.checkpoint_path.name}: "
                f"{loaded.beta}. Attendu : {expected_beta}."
            )

        if (
            loaded.checkpoint_split_seed is not None
            and loaded.checkpoint_split_seed
            != split_seed_protocol
        ):
            raise ValueError(
                "split_seed incompatible avec le protocole final "
                f"dans {loaded.checkpoint_path.name}: "
                f"{loaded.checkpoint_split_seed}. "
                f"Attendu : {split_seed_protocol}."
            )

        # Le checkpoint historique seed=42 a été entraîné avant
        # l'ajout explicite de split_seed. Son absence est acceptée,
        # car ce run a déjà été audité comme split_seed=42.
        if (
            loaded.checkpoint_split_seed is None
            and loaded.training_seed != 42
        ):
            raise ValueError(
                "split_seed absent dans un checkpoint récent : "
                f"{loaded.checkpoint_path.name}."
            )

    latent_dims = {
        loaded.latent_dim
        for loaded in loaded_models
    }

    hidden_dims = {
        loaded.hidden_dim
        for loaded in loaded_models
    }

    num_classes_values = {
        loaded.num_classes
        for loaded in loaded_models
    }

    if len(latent_dims) != 1:
        raise ValueError(
            "Les CVAE n'ont pas la même dimension latente : "
            f"{sorted(latent_dims)}."
        )

    if len(hidden_dims) != 1:
        raise ValueError(
            "Les CVAE n'ont pas la même dimension cachée : "
            f"{sorted(hidden_dims)}."
        )

    if num_classes_values != {
        len(FASHION_CLASSES)
    }:
        raise ValueError(
            "Le nombre de classes n'est pas compatible "
            "avec Fashion-MNIST : "
            f"{sorted(num_classes_values)}."
        )

    by_seed = {
        loaded.training_seed: loaded
        for loaded in loaded_models
    }

    return [
        by_seed[
            seed
        ]
        for seed in expected_seeds
    ]


# ============================================================
# CHARGEMENT DU CLASSIFIEUR
# ============================================================

def instantiate_classifier() -> FashionMNISTClassifier:
    """
    Construire le classifieur Fashion-MNIST indépendant.
    """

    return FashionMNISTClassifier(
        num_classes=10,
    )


def load_classifier(
    path: Path,
    device: torch.device,
) -> tuple[
    FashionMNISTClassifier,
    float | None,
]:
    """
    Charger le classifieur indépendant.
    """

    checkpoint_path = (
        resolve_project_path(
            path
        )
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_path,
        device,
    )

    model = instantiate_classifier()

    state_dict = strip_module_prefix(
        extract_state_dict(
            checkpoint
        )
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    validation_accuracy = (
        nested_checkpoint_value(
            checkpoint,
            [
                "validation_accuracy",
                "best_validation_accuracy",
                "val_accuracy",
            ],
            None,
        )
    )

    if validation_accuracy is not None:
        validation_accuracy = float(
            validation_accuracy
        )

    return (
        model,
        validation_accuracy,
    )


# ============================================================
# EXTRACTION DES CARACTÉRISTIQUES PÉNULTIÈMES
# ============================================================

class PenultimateFeatureExtractor:
    """
    Capturer l'entrée de la dernière couche Linear du classifieur.

    Dans le classifieur du projet, cette représentation possède
    128 dimensions.
    """

    def __init__(
        self,
        classifier: nn.Module,
    ) -> None:

        self.classifier = (
            classifier
        )

        self._latest_features: (
            torch.Tensor
            | None
        ) = None

        linear_layers = [
            module
            for module in classifier.modules()
            if isinstance(
                module,
                nn.Linear,
            )
        ]

        if not linear_layers:
            raise RuntimeError(
                "Le classifieur ne contient aucune couche nn.Linear."
            )

        self.final_linear = (
            linear_layers[
                -1
            ]
        )

        self._hook = (
            self.final_linear
            .register_forward_pre_hook(
                self._capture
            )
        )

    def _capture(
        self,
        module: nn.Module,
        inputs: tuple[
            torch.Tensor,
            ...
        ],
    ) -> None:
        """
        Capturer les features juste avant la couche finale.
        """

        del module

        if not inputs:
            raise RuntimeError(
                "Impossible de capturer les caractéristiques "
                "pénultièmes."
            )

        self._latest_features = (
            inputs[
                0
            ].detach()
        )

    @torch.no_grad()
    def predict_with_features(
        self,
        images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retourner les logits et les caractéristiques pénultièmes.
        """

        self._latest_features = None

        logits = self.classifier(
            images
        )

        if self._latest_features is None:
            raise RuntimeError(
                "Le hook n'a capturé aucune caractéristique."
            )

        return (
            logits,
            self._latest_features,
        )

    def close(self) -> None:
        """
        Supprimer proprement le hook.
        """

        self._hook.remove()


# ============================================================
# BANQUE LATENTE CONTRÔLÉE
# ============================================================

def create_shared_latent_bank(
    samples_per_class: int,
    latent_dim: int,
    seed: int,
) -> torch.Tensor:
    """
    Créer une seule banque z sur CPU, commune aux trois seeds
    et aux dix classes.
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    return torch.randn(
        samples_per_class,
        latent_dim,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


# ============================================================
# MÉTRIQUES DE DIVERSITÉ
# ============================================================

def pairwise_pixel_rms(
    samples: torch.Tensor,
) -> float:
    """
    Calculer un RMS pixel global sur toutes les paires.

    L'identité de variance évite de construire une matrice n x n :

        somme_{i<j} ||x_i - x_j||²
        =
        n * somme_i ||x_i - moyenne||²

    La métrique retournée correspond à la racine de la moyenne
    des différences pixel² sur toutes les paires et dimensions.
    """

    if samples.ndim != 2:
        samples = samples.flatten(
            start_dim=1
        )

    n, d = samples.shape

    if n < 2 or d < 1:
        return float(
            "nan"
        )

    samples64 = samples.to(
        dtype=torch.float64
    )

    mean = samples64.mean(
        dim=0,
        keepdim=True,
    )

    centered = (
        samples64
        - mean
    )

    sum_squared_deviation = (
        centered
        .pow(2)
        .sum()
    )

    mean_pairwise_squared_per_dimension = (
        2.0
        * sum_squared_deviation
        / (
            (n - 1)
            * d
        )
    )

    rms = torch.sqrt(
        mean_pairwise_squared_per_dimension
        .clamp_min(
            0.0
        )
    )

    return float(
        rms.item()
    )


def pairwise_feature_cosine_distance(
    features: torch.Tensor,
) -> float:
    """
    Calculer exactement la distance cosinus moyenne entre toutes
    les paires, sans construire une matrice n x n.
    """

    if features.ndim != 2:
        features = features.flatten(
            start_dim=1
        )

    n = features.shape[
        0
    ]

    if n < 2:
        return float(
            "nan"
        )

    normalized = F.normalize(
        features.to(
            dtype=torch.float64
        ),
        p=2,
        dim=1,
        eps=1e-12,
    )

    summed = normalized.sum(
        dim=0
    )

    mean_pairwise_similarity = (
        summed.dot(
            summed
        )
        - n
    ) / (
        n
        * (n - 1)
    )

    mean_pairwise_distance = (
        1.0
        - mean_pairwise_similarity
    )

    mean_pairwise_distance = (
        mean_pairwise_distance
        .clamp(
            min=0.0,
            max=2.0,
        )
    )

    return float(
        mean_pairwise_distance.item()
    )


def finite_mean(
    values: Iterable[float],
) -> float:
    """
    Moyenne des valeurs finies.
    """

    finite_values = [
        float(
            value
        )
        for value in values
        if math.isfinite(
            float(
                value
            )
        )
    ]

    if not finite_values:
        return float(
            "nan"
        )

    return float(
        np.mean(
            finite_values
        )
    )


def sample_std(
    values: Iterable[float],
) -> float:
    """
    Écart-type échantillonnal, ddof=1.

    Retourne NaN si moins de deux valeurs finies sont disponibles.
    """

    finite_values = np.asarray(
        [
            float(
                value
            )
            for value in values
            if math.isfinite(
                float(
                    value
                )
            )
        ],
        dtype=np.float64,
    )

    if finite_values.size < 2:
        return float(
            "nan"
        )

    return float(
        np.std(
            finite_values,
            ddof=1,
        )
    )


def finite_min(
    values: Iterable[float],
) -> float:
    """
    Minimum des valeurs finies.
    """

    finite_values = [
        float(
            value
        )
        for value in values
        if math.isfinite(
            float(
                value
            )
        )
    ]

    if not finite_values:
        return float(
            "nan"
        )

    return float(
        min(
            finite_values
        )
    )


def finite_max(
    values: Iterable[float],
) -> float:
    """
    Maximum des valeurs finies.
    """

    finite_values = [
        float(
            value
        )
        for value in values
        if math.isfinite(
            float(
                value
            )
        )
    ]

    if not finite_values:
        return float(
            "nan"
        )

    return float(
        max(
            finite_values
        )
    )


# ============================================================
# ÉVALUATION D'UN CVAE
# ============================================================

@torch.inference_mode()
def evaluate_one_cvae(
    loaded: LoadedCVAE,
    classifier_features: PenultimateFeatureExtractor,
    latent_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
    split_seed_protocol: int,
    latent_seed: int,
) -> tuple[
    dict[str, Any],
    list[
        dict[str, Any]
    ],
]:
    """
    Évaluer la diversité intra-classe d'un checkpoint CVAE.
    """

    model = loaded.model

    samples_per_class = (
        latent_bank.shape[
            0
        ]
    )

    per_class_rows: list[
        dict[str, Any]
    ] = []

    total_correct = 0
    total_generated = 0

    for class_id, class_name in enumerate(
        FASHION_CLASSES
    ):

        pixel_chunks: list[
            torch.Tensor
        ] = []

        feature_chunks: list[
            torch.Tensor
        ] = []

        prediction_chunks: list[
            torch.Tensor
        ] = []

        for start in range(
            0,
            samples_per_class,
            batch_size,
        ):

            end = min(
                start
                + batch_size,
                samples_per_class,
            )

            z = (
                latent_bank[
                    start:end
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            labels = torch.full(
                size=(
                    end - start,
                ),
                fill_value=(
                    class_id
                ),
                dtype=torch.long,
                device=device,
            )

            generated = model.decode(
                z,
                labels,
            )

            if generated.ndim == 2:
                generated = (
                    generated.view(
                        generated.shape[
                            0
                        ],
                        1,
                        28,
                        28,
                    )
                )

            generated = generated.clamp(
                0.0,
                1.0,
            )

            logits, features = (
                classifier_features
                .predict_with_features(
                    generated
                )
            )

            predictions = logits.argmax(
                dim=1
            )

            pixel_chunks.append(
                generated
                .detach()
                .flatten(
                    start_dim=1
                )
                .cpu()
            )

            feature_chunks.append(
                features
                .detach()
                .flatten(
                    start_dim=1
                )
                .cpu()
            )

            prediction_chunks.append(
                predictions
                .detach()
                .cpu()
            )

        pixels = torch.cat(
            pixel_chunks,
            dim=0,
        )

        features = torch.cat(
            feature_chunks,
            dim=0,
        )

        predictions = torch.cat(
            prediction_chunks,
            dim=0,
        )

        coherent_mask = (
            predictions.eq(
                class_id
            )
        )

        coherent_count = int(
            coherent_mask
            .sum()
            .item()
        )

        total_correct += (
            coherent_count
        )

        total_generated += (
            samples_per_class
        )

        pixel_all = pairwise_pixel_rms(
            pixels
        )

        feature_all = (
            pairwise_feature_cosine_distance(
                features
            )
        )

        if coherent_count >= 2:

            pixel_coherent = pairwise_pixel_rms(
                pixels[
                    coherent_mask
                ]
            )

            feature_coherent = (
                pairwise_feature_cosine_distance(
                    features[
                        coherent_mask
                    ]
                )
            )

        else:

            pixel_coherent = float(
                "nan"
            )

            feature_coherent = float(
                "nan"
            )

        conditional_accuracy = (
            coherent_count
            / samples_per_class
        )

        per_class_rows.append(
            {
                "checkpoint": (
                    loaded
                    .checkpoint_path
                    .name
                ),
                "model_type": (
                    "CVAE"
                ),
                "beta": (
                    loaded.beta
                ),
                "training_seed": (
                    loaded.training_seed
                ),
                "split_seed_protocol": (
                    split_seed_protocol
                ),
                "checkpoint_split_seed": (
                    loaded.checkpoint_split_seed
                ),
                "latent_seed": (
                    latent_seed
                ),
                "best_epoch": (
                    loaded.best_epoch
                ),
                "class_id": (
                    class_id
                ),
                "class_name": (
                    class_name
                ),
                "generated_count": (
                    samples_per_class
                ),
                "coherent_count": (
                    coherent_count
                ),
                "conditional_accuracy": (
                    conditional_accuracy
                ),
                "pixel_pairwise_rms_all": (
                    pixel_all
                ),
                "pixel_pairwise_rms_coherent": (
                    pixel_coherent
                ),
                "feature_cosine_diversity_all": (
                    feature_all
                ),
                "feature_cosine_diversity_coherent": (
                    feature_coherent
                ),
            }
        )

    summary = {
        "checkpoint": (
            loaded
            .checkpoint_path
            .name
        ),
        "model_type": (
            "CVAE"
        ),
        "beta": (
            loaded.beta
        ),
        "training_seed": (
            loaded.training_seed
        ),
        "split_seed_protocol": (
            split_seed_protocol
        ),
        "checkpoint_split_seed": (
            loaded.checkpoint_split_seed
        ),
        "latent_seed": (
            latent_seed
        ),
        "best_epoch": (
            loaded.best_epoch
        ),
        "latent_dim": (
            loaded.latent_dim
        ),
        "generated_count": (
            total_generated
        ),
        "coherent_count": (
            total_correct
        ),
        "conditional_accuracy": (
            total_correct
            / total_generated
        ),
        "pixel_pairwise_rms_all_macro": (
            finite_mean(
                row[
                    "pixel_pairwise_rms_all"
                ]
                for row in per_class_rows
            )
        ),
        "pixel_pairwise_rms_coherent_macro": (
            finite_mean(
                row[
                    "pixel_pairwise_rms_coherent"
                ]
                for row in per_class_rows
            )
        ),
        "feature_cosine_diversity_all_macro": (
            finite_mean(
                row[
                    "feature_cosine_diversity_all"
                ]
                for row in per_class_rows
            )
        ),
        "feature_cosine_diversity_coherent_macro": (
            finite_mean(
                row[
                    "feature_cosine_diversity_coherent"
                ]
                for row in per_class_rows
            )
        ),
    }

    return (
        summary,
        per_class_rows,
    )


# ============================================================
# AGRÉGATION MULTI-SEED
# ============================================================

def build_multiseed_summary(
    summary_rows: Sequence[
        dict[str, Any]
    ],
    expected_beta: float,
    expected_training_seeds: Sequence[int],
    split_seed_protocol: int,
    latent_seed: int,
) -> dict[str, Any]:
    """
    Agréger les métriques globales sur les seeds d'entraînement.
    """

    if not summary_rows:
        raise ValueError(
            "Aucun résumé individuel à agréger."
        )

    metric_names = [
        "conditional_accuracy",
        "pixel_pairwise_rms_all_macro",
        "pixel_pairwise_rms_coherent_macro",
        "feature_cosine_diversity_all_macro",
        "feature_cosine_diversity_coherent_macro",
    ]

    aggregated: dict[
        str,
        Any
    ] = {
        "model_type": (
            "CVAE"
        ),
        "beta": (
            expected_beta
        ),
        "n_seeds": (
            len(
                summary_rows
            )
        ),
        "training_seeds": (
            ",".join(
                str(
                    seed
                )
                for seed in expected_training_seeds
            )
        ),
        "split_seed_protocol": (
            split_seed_protocol
        ),
        "latent_seed": (
            latent_seed
        ),
        "generated_images_per_seed": int(
            summary_rows[
                0
            ][
                "generated_count"
            ]
        ),
        "total_generated_images_across_seeds": int(
            sum(
                int(
                    row[
                        "generated_count"
                    ]
                )
                for row in summary_rows
            )
        ),
        "total_coherent_images_across_seeds": int(
            sum(
                int(
                    row[
                        "coherent_count"
                    ]
                )
                for row in summary_rows
            )
        ),
    }

    total_generated = (
        aggregated[
            "total_generated_images_across_seeds"
        ]
    )

    total_coherent = (
        aggregated[
            "total_coherent_images_across_seeds"
        ]
    )

    aggregated[
        "conditional_accuracy_pooled"
    ] = (
        total_coherent
        / total_generated
    )

    for metric_name in metric_names:

        values = [
            float(
                row[
                    metric_name
                ]
            )
            for row in summary_rows
        ]

        aggregated[
            f"{metric_name}_mean"
        ] = finite_mean(
            values
        )

        aggregated[
            f"{metric_name}_std"
        ] = sample_std(
            values
        )

        aggregated[
            f"{metric_name}_min"
        ] = finite_min(
            values
        )

        aggregated[
            f"{metric_name}_max"
        ] = finite_max(
            values
        )

    return aggregated


def build_per_class_multiseed_summary(
    per_class_rows: Sequence[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Agréger les métriques de chaque classe sur les seeds.
    """

    output_rows: list[
        dict[str, Any]
    ] = []

    metric_names = [
        "conditional_accuracy",
        "pixel_pairwise_rms_all",
        "pixel_pairwise_rms_coherent",
        "feature_cosine_diversity_all",
        "feature_cosine_diversity_coherent",
    ]

    for class_id, class_name in enumerate(
        FASHION_CLASSES
    ):

        class_rows = [
            row
            for row in per_class_rows
            if int(
                row[
                    "class_id"
                ]
            ) == class_id
        ]

        if not class_rows:
            raise RuntimeError(
                f"Aucun résultat pour la classe {class_id}."
            )

        result: dict[
            str,
            Any
        ] = {
            "class_id": (
                class_id
            ),
            "class_name": (
                class_name
            ),
            "n_seeds": (
                len(
                    class_rows
                )
            ),
        }

        for metric_name in metric_names:

            values = [
                float(
                    row[
                        metric_name
                    ]
                )
                for row in class_rows
            ]

            result[
                f"{metric_name}_mean"
            ] = finite_mean(
                values
            )

            result[
                f"{metric_name}_std"
            ] = sample_std(
                values
            )

            result[
                f"{metric_name}_min"
            ] = finite_min(
                values
            )

            result[
                f"{metric_name}_max"
            ] = finite_max(
                values
            )

        output_rows.append(
            result
        )

    return output_rows


# ============================================================
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: Sequence[
        dict[str, Any]
    ],
) -> None:
    """
    Sauvegarder une séquence de dictionnaires dans un CSV.
    """

    rows = list(
        rows
    )

    if not rows:
        raise ValueError(
            f"Aucune ligne à écrire dans {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# FIGURES
# ============================================================

def save_metric_by_seed_plot(
    summary_rows: Sequence[
        dict[str, Any]
    ],
    metric_name: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Tracer une métrique globale pour chaque training_seed.
    """

    rows = sorted(
        summary_rows,
        key=lambda row: int(
            row[
                "training_seed"
            ]
        ),
    )

    labels = [
        f"seed {int(row['training_seed'])}"
        for row in rows
    ]

    values = [
        float(
            row[
                metric_name
            ]
        )
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    axis.bar(
        labels,
        values,
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Seed d'entraînement"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_per_class_mean_std_plot(
    per_class_summary_rows: Sequence[
        dict[str, Any]
    ],
    mean_field: str,
    std_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Tracer moyenne ± écart-type multi-seed par classe.
    """

    rows = sorted(
        per_class_summary_rows,
        key=lambda row: int(
            row[
                "class_id"
            ]
        ),
    )

    x = np.arange(
        len(
            rows
        )
    )

    means = np.asarray(
        [
            float(
                row[
                    mean_field
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    stds = np.asarray(
        [
            float(
                row[
                    std_field
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )

    labels = [
        str(
            row[
                "class_name"
            ]
        )
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    axis.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linestyle="-",
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Classe demandée"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# MLFLOW
# ============================================================

def local_sqlite_tracking_uri() -> str:
    """
    Construire l'URI SQLite MLflow locale.
    """

    db_path = (
        PROJECT_ROOT
        / "mlflow.db"
    ).resolve()

    return (
        f"sqlite:///"
        f"{db_path.as_posix()}"
    )


def configure_mlflow(
    tracking_uri: str | None,
    experiment_name: str,
) -> tuple[
    Any,
    str,
    str,
]:
    """
    Configurer MLflow avec le backend SQLite du projet.
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    uri = (
        tracking_uri
        if tracking_uri
        else local_sqlite_tracking_uri()
    )

    mlflow.set_tracking_uri(
        uri
    )

    existing = (
        mlflow.get_experiment_by_name(
            experiment_name
        )
    )

    if existing is None:

        artifact_dir = (
            PROJECT_ROOT
            / "mlartifacts"
            / experiment_name.replace(
                "/",
                "_",
            )
        ).resolve()

        artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        client = MlflowClient()

        experiment_id = (
            client.create_experiment(
                name=(
                    experiment_name
                ),
                artifact_location=(
                    artifact_dir.as_uri()
                ),
            )
        )

    else:

        experiment_id = (
            existing.experiment_id
        )

    return (
        mlflow,
        experiment_id,
        uri,
    )


def seed_metric_token(
    seed: int,
) -> str:
    """
    Formater une seed pour les noms de métriques MLflow.
    """

    return str(
        seed
    ).replace(
        "-",
        "m",
    )


def log_results_to_mlflow(
    mlflow: Any,
    args: argparse.Namespace,
    device: torch.device,
    classifier_validation_accuracy: float | None,
    summary_rows: Sequence[
        dict[str, Any]
    ],
    per_class_rows: Sequence[
        dict[str, Any]
    ],
    multiseed_summary: dict[
        str,
        Any
    ],
    output_dir: Path,
    duration_seconds: float,
) -> None:
    """
    Journaliser les résultats individuels et multi-seed dans MLflow.
    """

    mlflow.log_params(
        {
            "evaluation": (
                "generation_diversity_multiseed"
            ),
            "dataset": (
                "Fashion-MNIST"
            ),
            "model_type": (
                "CVAE"
            ),
            "expected_beta": (
                args.expected_beta
            ),
            "expected_training_seeds": (
                ",".join(
                    str(
                        seed
                    )
                    for seed in args.expected_training_seeds
                )
            ),
            "split_seed_protocol": (
                args.split_seed_protocol
            ),
            "samples_per_class": (
                args.samples_per_class
            ),
            "generation_batch_size": (
                args.generation_batch_size
            ),
            "latent_seed": (
                args.seed
            ),
            "num_classes": (
                len(
                    FASHION_CLASSES
                )
            ),
            "num_cvae": (
                len(
                    summary_rows
                )
            ),
            "device": (
                str(
                    device
                )
            ),
            "classifier_checkpoint": (
                Path(
                    args.classifier_checkpoint
                ).name
            ),
            "shared_z_between_cvae": (
                True
            ),
            "shared_z_between_classes": (
                True
            ),
            "official_test_used": (
                False
            ),
            "pixel_metric": (
                "global_pairwise_pixel_rms"
            ),
            "feature_metric": (
                "mean_pairwise_cosine_distance"
            ),
            "feature_source": (
                "penultimate_classifier_representation"
            ),
        }
    )

    mlflow.set_tags(
        {
            "dataset": (
                "Fashion-MNIST"
            ),
            "evaluation_type": (
                "generation_diversity_multiseed"
            ),
            "comparison_control": (
                "shared_latent_vectors"
            ),
            "official_test_used": (
                "False"
            ),
        }
    )

    if classifier_validation_accuracy is not None:

        mlflow.log_param(
            "classifier_validation_accuracy",
            classifier_validation_accuracy,
        )

    for summary in summary_rows:

        seed = int(
            summary[
                "training_seed"
            ]
        )

        token = seed_metric_token(
            seed
        )

        mlflow.log_metrics(
            {
                (
                    f"conditional_accuracy_seed_{token}"
                ): float(
                    summary[
                        "conditional_accuracy"
                    ]
                ),
                (
                    f"pixel_diversity_all_seed_{token}"
                ): float(
                    summary[
                        "pixel_pairwise_rms_all_macro"
                    ]
                ),
                (
                    f"pixel_diversity_coherent_seed_{token}"
                ): float(
                    summary[
                        "pixel_pairwise_rms_coherent_macro"
                    ]
                ),
                (
                    f"feature_diversity_all_seed_{token}"
                ): float(
                    summary[
                        "feature_cosine_diversity_all_macro"
                    ]
                ),
                (
                    f"feature_diversity_coherent_seed_{token}"
                ): float(
                    summary[
                        "feature_cosine_diversity_coherent_macro"
                    ]
                ),
            }
        )

    for row in per_class_rows:

        seed = int(
            row[
                "training_seed"
            ]
        )

        class_id = int(
            row[
                "class_id"
            ]
        )

        token = seed_metric_token(
            seed
        )

        metrics = {
            (
                f"conditional_accuracy_seed_"
                f"{token}_class_{class_id}"
            ): float(
                row[
                    "conditional_accuracy"
                ]
            ),
            (
                f"pixel_diversity_coherent_seed_"
                f"{token}_class_{class_id}"
            ): float(
                row[
                    "pixel_pairwise_rms_coherent"
                ]
            ),
            (
                f"feature_diversity_coherent_seed_"
                f"{token}_class_{class_id}"
            ): float(
                row[
                    "feature_cosine_diversity_coherent"
                ]
            ),
        }

        finite_metrics = {
            name: value
            for name, value in metrics.items()
            if math.isfinite(
                value
            )
        }

        if finite_metrics:

            mlflow.log_metrics(
                finite_metrics
            )

    aggregate_metric_names = [
        "conditional_accuracy",
        "pixel_pairwise_rms_all_macro",
        "pixel_pairwise_rms_coherent_macro",
        "feature_cosine_diversity_all_macro",
        "feature_cosine_diversity_coherent_macro",
    ]

    for metric_name in aggregate_metric_names:

        mean_value = float(
            multiseed_summary[
                f"{metric_name}_mean"
            ]
        )

        std_value = float(
            multiseed_summary[
                f"{metric_name}_std"
            ]
        )

        if math.isfinite(
            mean_value
        ):
            mlflow.log_metric(
                f"{metric_name}_mean",
                mean_value,
            )

        if math.isfinite(
            std_value
        ):
            mlflow.log_metric(
                f"{metric_name}_std",
                std_value,
            )

    mlflow.log_metric(
        "conditional_accuracy_pooled",
        float(
            multiseed_summary[
                "conditional_accuracy_pooled"
            ]
        ),
    )

    mlflow.log_metric(
        "duration_seconds",
        float(
            duration_seconds
        ),
    )

    mlflow.log_artifacts(
        str(
            output_dir
        ),
        artifact_path=(
            "generation_diversity_multiseed"
        ),
    )


# ============================================================
# AFFICHAGE
# ============================================================

def print_model_results(
    summary: dict[str, Any],
    per_class_rows: Sequence[
        dict[str, Any]
    ],
) -> None:
    """
    Afficher les résultats détaillés d'une seed.
    """

    print(
        "=" * 100
    )

    print(
        "CVAE : "
        f"{summary['checkpoint']}"
    )

    print(
        "=" * 100
    )

    print(
        "Training seed                     : "
        f"{summary['training_seed']}"
    )

    print(
        "Beta                              : "
        f"{summary['beta']:g}"
    )

    print(
        "Époque du checkpoint              : "
        f"{summary['best_epoch']}"
    )

    print(
        "Images générées                   : "
        f"{summary['generated_count']}"
    )

    print(
        "Conditional accuracy              : "
        f"{summary['conditional_accuracy']:.4%}"
    )

    print(
        "Diversité pixel ALL (macro)       : "
        f"{summary['pixel_pairwise_rms_all_macro']:.6f}"
    )

    print(
        "Diversité pixel COHERENT (macro)  : "
        f"{summary['pixel_pairwise_rms_coherent_macro']:.6f}"
    )

    print(
        "Diversité feature ALL (macro)     : "
        f"{summary['feature_cosine_diversity_all_macro']:.6f}"
    )

    print(
        "Diversité feature COHERENT (macro): "
        f"{summary['feature_cosine_diversity_coherent_macro']:.6f}"
    )

    print(
        "-" * 100
    )

    print(
        f"{'ID':>2}  "
        f"{'Classe':<13} "
        f"{'Cohérence':>10}  "
        f"{'Pixel coh.':>11}  "
        f"{'Feature coh.':>13}"
    )

    print(
        "-" * 100
    )

    for row in per_class_rows:

        print(
            f"{int(row['class_id']):>2}  "
            f"{str(row['class_name']):<13} "
            f"{float(row['conditional_accuracy']):>9.2%}  "
            f"{float(row['pixel_pairwise_rms_coherent']):>11.6f}  "
            f"{float(row['feature_cosine_diversity_coherent']):>13.6f}"
        )

    print(
        "=" * 100
    )


def print_multiseed_summary(
    summary_rows: Sequence[
        dict[str, Any]
    ],
    multiseed_summary: dict[
        str,
        Any
    ],
) -> None:
    """
    Afficher les valeurs individuelles puis moyenne ± écart-type.
    """

    print(
        "\n"
        + "=" * 104
    )

    print(
        "DIVERSITÉ CVAE BETA=1 — RÉSULTATS PAR SEED"
    )

    print(
        "=" * 104
    )

    print(
        f"{'Seed':>6} "
        f"{'Cond. acc.':>12} "
        f"{'Pixel ALL':>12} "
        f"{'Pixel coh.':>12} "
        f"{'Feature ALL':>13} "
        f"{'Feature coh.':>13}"
    )

    print(
        "-" * 104
    )

    for row in summary_rows:

        print(
            f"{int(row['training_seed']):>6} "
            f"{float(row['conditional_accuracy']):>11.2%} "
            f"{float(row['pixel_pairwise_rms_all_macro']):>12.6f} "
            f"{float(row['pixel_pairwise_rms_coherent_macro']):>12.6f} "
            f"{float(row['feature_cosine_diversity_all_macro']):>13.6f} "
            f"{float(row['feature_cosine_diversity_coherent_macro']):>13.6f}"
        )

    print(
        "=" * 104
    )

    print(
        "Moyenne ± écart-type (n="
        f"{multiseed_summary['n_seeds']})"
    )

    print(
        "Conditional accuracy              : "
        f"{multiseed_summary['conditional_accuracy_mean']:.4%} "
        "± "
        f"{multiseed_summary['conditional_accuracy_std']:.4%}"
    )

    print(
        "Diversité pixel ALL               : "
        f"{multiseed_summary['pixel_pairwise_rms_all_macro_mean']:.6f} "
        "± "
        f"{multiseed_summary['pixel_pairwise_rms_all_macro_std']:.6f}"
    )

    print(
        "Diversité pixel COHERENT          : "
        f"{multiseed_summary['pixel_pairwise_rms_coherent_macro_mean']:.6f} "
        "± "
        f"{multiseed_summary['pixel_pairwise_rms_coherent_macro_std']:.6f}"
    )

    print(
        "Diversité feature ALL             : "
        f"{multiseed_summary['feature_cosine_diversity_all_macro_mean']:.6f} "
        "± "
        f"{multiseed_summary['feature_cosine_diversity_all_macro_std']:.6f}"
    )

    print(
        "Diversité feature COHERENT        : "
        f"{multiseed_summary['feature_cosine_diversity_coherent_macro_mean']:.6f} "
        "± "
        f"{multiseed_summary['feature_cosine_diversity_coherent_macro_std']:.6f}"
    )

    print(
        "Conditional accuracy pooled       : "
        f"{multiseed_summary['conditional_accuracy_pooled']:.4%}"
    )

    print(
        "=" * 104
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    """
    Point d'entrée principal.
    """

    args = parse_args()

    start_time = (
        time.perf_counter()
    )

    set_random_seed(
        args.seed
    )

    device = select_device(
        args.device
    )

    output_dir = (
        resolve_project_path(
            args.output_dir
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier_checkpoint_path = (
        resolve_project_path(
            args.classifier_checkpoint
        )
    )

    # --------------------------------------------------------
    # Classifieur indépendant
    # --------------------------------------------------------

    (
        classifier,
        classifier_validation_accuracy,
    ) = load_classifier(
        classifier_checkpoint_path,
        device,
    )

    classifier_features = (
        PenultimateFeatureExtractor(
            classifier
        )
    )

    # --------------------------------------------------------
    # CVAE
    # --------------------------------------------------------

    loaded_models = [
        load_cvae_checkpoint(
            path,
            device,
        )
        for path in args.checkpoints
    ]

    loaded_models = validate_loaded_models(
        loaded_models=(
            loaded_models
        ),
        expected_beta=(
            args.expected_beta
        ),
        expected_training_seeds=(
            args.expected_training_seeds
        ),
        split_seed_protocol=(
            args.split_seed_protocol
        ),
    )

    latent_dim = (
        loaded_models[
            0
        ].latent_dim
    )

    # --------------------------------------------------------
    # Banque z commune
    # --------------------------------------------------------

    latent_bank = create_shared_latent_bank(
        samples_per_class=(
            args.samples_per_class
        ),
        latent_dim=(
            latent_dim
        ),
        seed=(
            args.seed
        ),
    )

    # --------------------------------------------------------
    # Informations générales
    # --------------------------------------------------------

    print(
        "=" * 104
    )

    print(
        "ÉVALUATION MULTI-SEED CONTRÔLÉE DE LA "
        "DIVERSITÉ DES GÉNÉRATIONS CVAE"
    )

    print(
        "=" * 104
    )

    print(
        "Appareil utilisé                  : "
        f"{device}"
    )

    print(
        "Classifieur                       : "
        f"{classifier_checkpoint_path.name}"
    )

    if classifier_validation_accuracy is not None:

        print(
            "Accuracy validation classifieur   : "
            f"{classifier_validation_accuracy:.4%}"
        )

    print(
        "Beta attendu                      : "
        f"{args.expected_beta:g}"
    )

    print(
        "Seeds d'entraînement              : "
        + ", ".join(
            str(
                seed
            )
            for seed in args.expected_training_seeds
        )
    )

    print(
        "Seed du split protocole           : "
        f"{args.split_seed_protocol}"
    )

    print(
        "Nombre de CVAE                    : "
        f"{len(loaded_models)}"
    )

    print(
        "Dimension latente                 : "
        f"{latent_dim}"
    )

    print(
        "Images par classe et par seed     : "
        f"{args.samples_per_class}"
    )

    print(
        "Images générées par CVAE          : "
        f"{args.samples_per_class * len(FASHION_CLASSES)}"
    )

    print(
        "Images générées au total          : "
        f"{args.samples_per_class * len(FASHION_CLASSES) * len(loaded_models)}"
    )

    print(
        "Seed des vecteurs latents         : "
        f"{args.seed}"
    )

    print(
        "Mêmes z entre les seeds           : OUI"
    )

    print(
        "Mêmes z entre les classes         : OUI"
    )

    print(
        "Jeu officiel de test utilisé      : NON"
    )

    print(
        "=" * 104
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow_module = None

    run_context = nullcontext(
        None
    )

    if not args.disable_mlflow:

        (
            mlflow_module,
            experiment_id,
            tracking_uri,
        ) = configure_mlflow(
            tracking_uri=(
                args.mlflow_tracking_uri
            ),
            experiment_name=(
                args.mlflow_experiment_name
            ),
        )

        run_context = (
            mlflow_module.start_run(
                experiment_id=(
                    experiment_id
                ),
                run_name=(
                    args.mlflow_run_name
                ),
            )
        )

        print(
            "MLflow tracking URI               : "
            f"{tracking_uri}"
        )

        print(
            "=" * 104
        )

    summary_rows: list[
        dict[str, Any]
    ] = []

    all_per_class_rows: list[
        dict[str, Any]
    ] = []

    try:

        with run_context as active_run:

            if active_run is not None:

                print(
                    "MLflow run ID                     : "
                    f"{active_run.info.run_id}"
                )

                print(
                    "=" * 104
                )

            # ------------------------------------------------
            # Évaluation des trois seeds
            # ------------------------------------------------

            for loaded in loaded_models:

                print(
                    "\nÉvaluation de "
                    f"{loaded.checkpoint_path.name}..."
                )

                (
                    summary,
                    per_class_rows,
                ) = evaluate_one_cvae(
                    loaded=(
                        loaded
                    ),
                    classifier_features=(
                        classifier_features
                    ),
                    latent_bank=(
                        latent_bank
                    ),
                    batch_size=(
                        args.generation_batch_size
                    ),
                    device=(
                        device
                    ),
                    split_seed_protocol=(
                        args.split_seed_protocol
                    ),
                    latent_seed=(
                        args.seed
                    ),
                )

                summary_rows.append(
                    summary
                )

                all_per_class_rows.extend(
                    per_class_rows
                )

                print_model_results(
                    summary=(
                        summary
                    ),
                    per_class_rows=(
                        per_class_rows
                    ),
                )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # ------------------------------------------------
            # Agrégation multi-seed
            # ------------------------------------------------

            multiseed_summary = (
                build_multiseed_summary(
                    summary_rows=(
                        summary_rows
                    ),
                    expected_beta=(
                        args.expected_beta
                    ),
                    expected_training_seeds=(
                        args.expected_training_seeds
                    ),
                    split_seed_protocol=(
                        args.split_seed_protocol
                    ),
                    latent_seed=(
                        args.seed
                    ),
                )
            )

            per_class_multiseed_summary = (
                build_per_class_multiseed_summary(
                    per_class_rows=(
                        all_per_class_rows
                    )
                )
            )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            runs_path = (
                output_dir
                / "generation_diversity_multiseed_runs.csv"
            )

            summary_path = (
                output_dir
                / "generation_diversity_multiseed_summary.csv"
            )

            per_class_path = (
                output_dir
                / "generation_diversity_multiseed_per_class.csv"
            )

            per_class_summary_path = (
                output_dir
                / "generation_diversity_multiseed_per_class_summary.csv"
            )

            write_csv(
                runs_path,
                summary_rows,
            )

            write_csv(
                summary_path,
                [
                    multiseed_summary
                ],
            )

            write_csv(
                per_class_path,
                all_per_class_rows,
            )

            write_csv(
                per_class_summary_path,
                per_class_multiseed_summary,
            )

            # ------------------------------------------------
            # FIGURES
            # ------------------------------------------------

            pixel_by_seed_path = (
                output_dir
                / "generation_diversity_pixel_coherent_by_seed.png"
            )

            feature_by_seed_path = (
                output_dir
                / "generation_diversity_feature_coherent_by_seed.png"
            )

            pixel_per_class_path = (
                output_dir
                / "generation_diversity_pixel_per_class_mean_std.png"
            )

            feature_per_class_path = (
                output_dir
                / "generation_diversity_feature_per_class_mean_std.png"
            )

            save_metric_by_seed_plot(
                summary_rows=(
                    summary_rows
                ),
                metric_name=(
                    "pixel_pairwise_rms_coherent_macro"
                ),
                title=(
                    "Diversité pixel cohérente du CVAE final "
                    "selon la seed d'entraînement"
                ),
                ylabel=(
                    "RMS pixel global paire-à-paire"
                ),
                output_path=(
                    pixel_by_seed_path
                ),
            )

            save_metric_by_seed_plot(
                summary_rows=(
                    summary_rows
                ),
                metric_name=(
                    "feature_cosine_diversity_coherent_macro"
                ),
                title=(
                    "Diversité sémantique cohérente du CVAE final "
                    "selon la seed d'entraînement"
                ),
                ylabel=(
                    "Distance cosinus moyenne paire-à-paire"
                ),
                output_path=(
                    feature_by_seed_path
                ),
            )

            save_per_class_mean_std_plot(
                per_class_summary_rows=(
                    per_class_multiseed_summary
                ),
                mean_field=(
                    "pixel_pairwise_rms_coherent_mean"
                ),
                std_field=(
                    "pixel_pairwise_rms_coherent_std"
                ),
                title=(
                    "Diversité pixel cohérente par classe "
                    "(moyenne ± écart-type, 3 seeds)"
                ),
                ylabel=(
                    "RMS pixel global paire-à-paire"
                ),
                output_path=(
                    pixel_per_class_path
                ),
            )

            save_per_class_mean_std_plot(
                per_class_summary_rows=(
                    per_class_multiseed_summary
                ),
                mean_field=(
                    "feature_cosine_diversity_coherent_mean"
                ),
                std_field=(
                    "feature_cosine_diversity_coherent_std"
                ),
                title=(
                    "Diversité sémantique cohérente par classe "
                    "(moyenne ± écart-type, 3 seeds)"
                ),
                ylabel=(
                    "Distance cosinus moyenne paire-à-paire"
                ),
                output_path=(
                    feature_per_class_path
                ),
            )

            duration_seconds = (
                time.perf_counter()
                - start_time
            )

            # ------------------------------------------------
            # MLFLOW
            # ------------------------------------------------

            if mlflow_module is not None:

                log_results_to_mlflow(
                    mlflow=(
                        mlflow_module
                    ),
                    args=(
                        args
                    ),
                    device=(
                        device
                    ),
                    classifier_validation_accuracy=(
                        classifier_validation_accuracy
                    ),
                    summary_rows=(
                        summary_rows
                    ),
                    per_class_rows=(
                        all_per_class_rows
                    ),
                    multiseed_summary=(
                        multiseed_summary
                    ),
                    output_dir=(
                        output_dir
                    ),
                    duration_seconds=(
                        duration_seconds
                    ),
                )

            # ------------------------------------------------
            # RÉSUMÉ FINAL
            # ------------------------------------------------

            print_multiseed_summary(
                summary_rows=(
                    summary_rows
                ),
                multiseed_summary=(
                    multiseed_summary
                ),
            )

            print(
                "Runs CSV                          : "
                f"{runs_path}"
            )

            print(
                "Résumé multi-seed CSV            : "
                f"{summary_path}"
            )

            print(
                "Résultats par classe             : "
                f"{per_class_path}"
            )

            print(
                "Résumé par classe                : "
                f"{per_class_summary_path}"
            )

            print(
                "Durée totale                     : "
                f"{duration_seconds:.2f} s"
            )

            print(
                "Jeu officiel de test             : "
                "NON UTILISÉ"
            )

            print(
                "=" * 104
            )

    finally:

        classifier_features.close()


if __name__ == "__main__":
    main()