"""
Évaluation contrôlée de la diversité des générations CVAE Fashion-MNIST.

Objectif
--------
Comparer la diversité des images générées par plusieurs CVAE, notamment
pour beta = 0.1, beta = 1 et beta = 4.

La comparaison est contrôlée :
- les mêmes vecteurs latents z sont utilisés pour tous les CVAE ;
- les mêmes vecteurs z sont également utilisés pour toutes les classes ;
- le jeu officiel de test Fashion-MNIST n'est jamais utilisé.

Deux familles de métriques sont calculées :

1. Diversité pixel
   RMS moyen des différences entre toutes les paires d'images d'une même
   classe.

2. Diversité sémantique
   Distance cosinus moyenne entre les représentations pénultièmes produites
   par le classifieur Fashion-MNIST indépendant.

Les métriques sont calculées :
- sur toutes les images générées ;
- uniquement sur les images jugées cohérentes avec la classe demandée par
  le classifieur indépendant.

Cette seconde mesure est importante : elle évite de considérer comme
"divers" un modèle qui produit surtout des images incohérentes ou hors classe.
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
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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
    PROJECT_ROOT / "checkpoints" / "cvae_beta_01_seed42_final.pt",
    PROJECT_ROOT / "checkpoints" / "cvae_beta_1_seed42_final.pt",
    PROJECT_ROOT / "checkpoints" / "cvae_beta_4_seed42_final.pt",
]

DEFAULT_CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "fashion_classifier_seed42_final.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "generation_diversity"
)


# ============================================================
# STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class LoadedCVAE:
    """
    Informations utiles associées à un CVAE chargé.
    """

    model: CVAE
    checkpoint_path: Path
    beta: float
    best_epoch: int | None
    latent_dim: int
    hidden_dim: int
    num_classes: int


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Lire les arguments de la ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Mesurer la diversité intra-classe des générations "
            "CVAE Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=DEFAULT_CVAE_CHECKPOINTS,
        help=(
            "Checkpoints CVAE à comparer. "
            "Par défaut : beta 0.1, 1 et 4."
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
        "--samples-per-class",
        type=int,
        default=1000,
        help=(
            "Nombre d'images générées par classe et par CVAE. "
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
            "Seed utilisée pour créer la banque latente commune. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
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
        help="Dossier de sauvegarde des résultats.",
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="fashion_mnist_generation_diversity",
        help="Nom de l'expérience MLflow.",
    )

    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default="cvae_generation_diversity_seed42_final",
        help="Nom du run MLflow.",
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

    if args.samples_per_class < 2:
        parser.error(
            "--samples-per-class doit être >= 2."
        )

    if args.generation_batch_size < 1:
        parser.error(
            "--generation-batch-size doit être >= 1."
        )

    return args


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
# CHARGEMENT DES CHECKPOINTS
# ============================================================

def load_torch_checkpoint(
    path: Path,
    device: torch.device,
) -> Any:
    """
    Charger un checkpoint PyTorch.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    return checkpoint


def nested_checkpoint_value(
    checkpoint: Any,
    names: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Rechercher une valeur dans plusieurs zones usuelles
    d'un checkpoint.
    """

    if not isinstance(checkpoint, dict):
        return default

    scopes = [checkpoint]

    for key in (
        "config",
        "model_config",
        "hyperparameters",
        "hparams",
        "params",
        "training_config",
    ):
        value = checkpoint.get(key)

        if isinstance(value, dict):
            scopes.append(value)

    for scope in scopes:
        for name in names:
            if name in scope:
                return scope[name]

    return default


def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    Extraire le state_dict depuis différents formats possibles.
    """

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
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
        candidate = checkpoint.get(key)

        if (
            isinstance(candidate, dict)
            and candidate
            and all(
                torch.is_tensor(value)
                for value in candidate.values()
            )
        ):
            return candidate

    if (
        checkpoint
        and all(
            torch.is_tensor(value)
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
    Retirer le préfixe 'module.' éventuellement ajouté
    par DataParallel.
    """

    if (
        state_dict
        and all(
            key.startswith("module.")
            for key in state_dict
        )
    ):
        return {
            key[len("module."):]: value
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
    Construire un CVAE en respectant la signature réelle
    de sa classe.
    """

    signature = inspect.signature(
        CVAE.__init__
    )

    parameters = signature.parameters

    kwargs: dict[str, Any] = {}

    if "latent_dim" in parameters:
        kwargs["latent_dim"] = latent_dim

    if "hidden_dim" in parameters:
        kwargs["hidden_dim"] = hidden_dim

    if "num_classes" in parameters:
        kwargs["num_classes"] = num_classes

    return CVAE(**kwargs)


def load_cvae_checkpoint(
    path: Path,
    device: torch.device,
) -> LoadedCVAE:
    """
    Charger un checkpoint CVAE avec ses métadonnées.
    """

    checkpoint = load_torch_checkpoint(
        path,
        device,
    )

    latent_dim = int(
        nested_checkpoint_value(
            checkpoint,
            ["latent_dim"],
            16,
        )
    )

    hidden_dim = int(
        nested_checkpoint_value(
            checkpoint,
            ["hidden_dim"],
            256,
        )
    )

    num_classes = int(
        nested_checkpoint_value(
            checkpoint,
            ["num_classes", "n_classes"],
            10,
        )
    )

    beta = float(
        nested_checkpoint_value(
            checkpoint,
            ["beta"],
            float("nan"),
        )
    )

    best_epoch_value = nested_checkpoint_value(
        checkpoint,
        ["best_epoch", "epoch"],
        None,
    )

    if best_epoch_value is None:
        best_epoch = None
    else:
        best_epoch = int(best_epoch_value)

    model = instantiate_cvae(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    state_dict = strip_module_prefix(
        extract_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return LoadedCVAE(
        model=model,
        checkpoint_path=Path(path),
        beta=beta,
        best_epoch=best_epoch,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )


# ============================================================
# CHARGEMENT DU CLASSIFIEUR
# ============================================================

def instantiate_classifier() -> FashionMNISTClassifier:
    """
    Construire le classifieur Fashion-MNIST indépendant.

    Le nom réel de la classe dans le projet est :
    FashionMNISTClassifier.
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

    checkpoint = load_torch_checkpoint(
        path,
        device,
    )

    model = instantiate_classifier()

    state_dict = strip_module_prefix(
        extract_state_dict(checkpoint)
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    validation_accuracy = nested_checkpoint_value(
        checkpoint,
        [
            "validation_accuracy",
            "best_validation_accuracy",
            "val_accuracy",
        ],
        None,
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

    Dans notre classifieur :

        Linear(3136 -> 128)
        ReLU
        Dropout
        Linear(128 -> 10)

    L'entrée de la dernière couche Linear est donc un vecteur
    de 128 caractéristiques.

    Ce vecteur représente l'image dans l'espace discriminatif
    appris par le classifieur.
    """

    def __init__(
        self,
        classifier: nn.Module,
    ) -> None:
        self.classifier = classifier

        self._latest_features: torch.Tensor | None = None

        linear_layers = [
            module
            for module in classifier.modules()
            if isinstance(module, nn.Linear)
        ]

        if not linear_layers:
            raise RuntimeError(
                "Le classifieur ne contient aucune couche nn.Linear."
            )

        self.final_linear = linear_layers[-1]

        self._hook = (
            self.final_linear.register_forward_pre_hook(
                self._capture
            )
        )

    def _capture(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
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

        self._latest_features = inputs[0].detach()

    @torch.no_grad()
    def predict_with_features(
        self,
        images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retourner simultanément :
        - les logits ;
        - les caractéristiques pénultièmes.
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
    Créer une banque de vecteurs z commune à tous les modèles.

    La banque est créée sur CPU pour rester reproductible,
    indépendamment du GPU utilisé ensuite.
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    latent_bank = torch.randn(
        samples_per_class,
        latent_dim,
        generator=generator,
        dtype=torch.float32,
    )

    return latent_bank


# ============================================================
# MÉTRIQUES DE DIVERSITÉ
# ============================================================

def pairwise_pixel_rms(
    samples: torch.Tensor,
) -> float:
    """
    Calculer la distance RMS pixel moyenne entre toutes les paires.

    On évite volontairement de construire une matrice n x n.

    Identité utilisée :

        somme_{i<j} ||x_i - x_j||²
        =
        n * somme_i ||x_i - moyenne||²

    Cela permet un calcul exact avec une complexité mémoire O(n*d)
    au lieu de O(n²).
    """

    if samples.ndim != 2:
        samples = samples.flatten(
            start_dim=1
        )

    n, d = samples.shape

    if n < 2 or d < 1:
        return float("nan")

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
        mean_pairwise_squared_per_dimension.clamp_min(
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
    Calculer la distance cosinus moyenne entre toutes les paires
    de représentations.

    Après normalisation L2, on peut calculer la moyenne exacte
    sans créer toutes les paires.
    """

    if features.ndim != 2:
        features = features.flatten(
            start_dim=1
        )

    n = features.shape[0]

    if n < 2:
        return float("nan")

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
        summed.dot(summed)
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
        mean_pairwise_distance.clamp(
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
    Calculer une moyenne en ignorant les NaN éventuels.
    """

    finite_values = [
        value
        for value in values
        if math.isfinite(value)
    ]

    if not finite_values:
        return float("nan")

    return float(
        np.mean(
            finite_values
        )
    )


# ============================================================
# FORMATAGE
# ============================================================

def beta_label(
    beta: float,
) -> str:
    """
    Formater beta pour l'affichage.
    """

    if math.isnan(beta):
        return "unknown"

    if float(beta).is_integer():
        return str(
            int(beta)
        )

    return f"{beta:g}"


def beta_metric_token(
    beta: float,
) -> str:
    """
    Formater beta pour les noms de métriques MLflow.
    """

    label = beta_label(
        beta
    )

    return (
        label
        .replace("-", "m")
        .replace(".", "_")
    )


# ============================================================
# ÉVALUATION D'UN CVAE
# ============================================================

@torch.no_grad()
def evaluate_one_cvae(
    loaded: LoadedCVAE,
    classifier_features: PenultimateFeatureExtractor,
    latent_bank: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Évaluer la diversité intra-classe d'un CVAE.
    """

    model = loaded.model

    samples_per_class = latent_bank.shape[0]

    per_class_rows: list[
        dict[str, Any]
    ] = []

    total_correct = 0
    total_generated = 0

    for class_id, class_name in enumerate(
        FASHION_CLASSES
    ):
        pixel_chunks: list[torch.Tensor] = []
        feature_chunks: list[torch.Tensor] = []
        prediction_chunks: list[torch.Tensor] = []

        for start in range(
            0,
            samples_per_class,
            batch_size,
        ):
            end = min(
                start + batch_size,
                samples_per_class,
            )

            z = (
                latent_bank[start:end]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            labels = torch.full(
                (
                    end - start,
                ),
                class_id,
                dtype=torch.long,
                device=device,
            )

            generated = model.decode(
                z,
                labels,
            )

            if generated.ndim == 2:
                generated = generated.view(
                    generated.shape[0],
                    1,
                    28,
                    28,
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
                .flatten(start_dim=1)
                .cpu()
            )

            feature_chunks.append(
                features
                .detach()
                .flatten(start_dim=1)
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

        coherent_mask = predictions.eq(
            class_id
        )

        coherent_count = int(
            coherent_mask.sum().item()
        )

        total_correct += coherent_count
        total_generated += samples_per_class

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
            pixel_coherent = float("nan")
            feature_coherent = float("nan")

        conditional_accuracy = (
            coherent_count
            / samples_per_class
        )

        row = {
            "checkpoint": (
                loaded.checkpoint_path.name
            ),
            "beta": loaded.beta,
            "best_epoch": loaded.best_epoch,
            "class_id": class_id,
            "class_name": class_name,
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

        per_class_rows.append(
            row
        )

    summary = {
        "checkpoint": (
            loaded.checkpoint_path.name
        ),
        "beta": loaded.beta,
        "best_epoch": loaded.best_epoch,
        "latent_dim": loaded.latent_dim,
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
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """
    Sauvegarder une liste de dictionnaires dans un CSV.
    """

    if not rows:
        raise ValueError(
            f"Aucune ligne à écrire dans {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# FIGURES
# ============================================================

def save_summary_feature_plot(
    summary_rows: list[
        dict[str, Any]
    ],
    output_path: Path,
) -> None:
    """
    Tracer la diversité sémantique cohérente moyenne.
    """

    labels = [
        "β="
        + beta_label(
            row["beta"]
        )
        for row in summary_rows
    ]

    values = [
        row[
            "feature_cosine_diversity_coherent_macro"
        ]
        for row in summary_rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Diversité sémantique intra-classe\n"
        "(générations cohérentes uniquement)"
    )

    ax.set_xlabel(
        "CVAE"
    )

    ax.set_ylabel(
        "Distance cosinus moyenne paire-à-paire"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_summary_pixel_plot(
    summary_rows: list[
        dict[str, Any]
    ],
    output_path: Path,
) -> None:
    """
    Tracer la diversité pixel cohérente moyenne.
    """

    labels = [
        "β="
        + beta_label(
            row["beta"]
        )
        for row in summary_rows
    ]

    values = [
        row[
            "pixel_pairwise_rms_coherent_macro"
        ]
        for row in summary_rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Diversité pixel intra-classe\n"
        "(générations cohérentes uniquement)"
    )

    ax.set_xlabel(
        "CVAE"
    )

    ax.set_ylabel(
        "RMS pixel moyen paire-à-paire"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        fig
    )


def save_per_class_feature_plot(
    per_class_rows: list[
        dict[str, Any]
    ],
    output_path: Path,
) -> None:
    """
    Tracer la diversité sémantique cohérente classe par classe.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    betas: list[float] = []

    for row in per_class_rows:
        beta = row["beta"]

        if beta not in betas:
            betas.append(
                beta
            )

    x = np.arange(
        len(FASHION_CLASSES)
    )

    for beta in betas:
        rows = [
            row
            for row in per_class_rows
            if row["beta"] == beta
        ]

        rows.sort(
            key=lambda row: row["class_id"]
        )

        values = [
            row[
                "feature_cosine_diversity_coherent"
            ]
            for row in rows
        ]

        ax.plot(
            x,
            values,
            marker="o",
            label=(
                "β="
                + beta_label(
                    beta
                )
            ),
        )

    ax.set_title(
        "Diversité sémantique par classe "
        "(générations cohérentes)"
    )

    ax.set_xlabel(
        "Classe demandée"
    )

    ax.set_ylabel(
        "Distance cosinus moyenne paire-à-paire"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        FASHION_CLASSES,
        rotation=35,
        ha="right",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(
        fig
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
    Configurer MLflow avec SQLite.
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        uri = tracking_uri
    else:
        uri = local_sqlite_tracking_uri()

    mlflow.set_tracking_uri(
        uri
    )

    existing = mlflow.get_experiment_by_name(
        experiment_name
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
                name=experiment_name,
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


def log_results_to_mlflow(
    mlflow: Any,
    args: argparse.Namespace,
    device: torch.device,
    classifier_validation_accuracy: float | None,
    summary_rows: list[
        dict[str, Any]
    ],
    per_class_rows: list[
        dict[str, Any]
    ],
    output_dir: Path,
    duration_seconds: float,
) -> None:
    """
    Journaliser les résultats dans MLflow.
    """

    mlflow.log_params(
        {
            "samples_per_class": (
                args.samples_per_class
            ),
            "generation_batch_size": (
                args.generation_batch_size
            ),
            "seed": args.seed,
            "num_classes": (
                len(FASHION_CLASSES)
            ),
            "num_cvae": (
                len(summary_rows)
            ),
            "device": str(device),
            "classifier_checkpoint": (
                Path(
                    args.classifier_checkpoint
                ).name
            ),
            "shared_z_between_cvae": True,
            "shared_z_between_classes": True,
            "official_test_used": False,
            "pixel_metric": (
                "mean_pairwise_rms_per_pixel"
            ),
            "feature_metric": (
                "mean_pairwise_cosine_distance"
            ),
            "feature_source": (
                "penultimate_classifier_representation"
            ),
        }
    )

    if classifier_validation_accuracy is not None:
        mlflow.log_param(
            "classifier_validation_accuracy",
            classifier_validation_accuracy,
        )

    for summary in summary_rows:
        token = beta_metric_token(
            float(summary["beta"])
        )

        mlflow.log_metrics(
            {
                (
                    f"conditional_accuracy_beta_{token}"
                ): summary[
                    "conditional_accuracy"
                ],
                (
                    f"pixel_diversity_all_beta_{token}"
                ): summary[
                    "pixel_pairwise_rms_all_macro"
                ],
                (
                    f"pixel_diversity_coherent_beta_{token}"
                ): summary[
                    "pixel_pairwise_rms_coherent_macro"
                ],
                (
                    f"feature_diversity_all_beta_{token}"
                ): summary[
                    "feature_cosine_diversity_all_macro"
                ],
                (
                    f"feature_diversity_coherent_beta_{token}"
                ): summary[
                    "feature_cosine_diversity_coherent_macro"
                ],
            }
        )

    for row in per_class_rows:
        token = beta_metric_token(
            float(row["beta"])
        )

        class_id = int(
            row["class_id"]
        )

        metrics = {
            (
                f"conditional_accuracy_beta_"
                f"{token}_class_{class_id}"
            ): row[
                "conditional_accuracy"
            ],
            (
                f"pixel_diversity_coherent_beta_"
                f"{token}_class_{class_id}"
            ): row[
                "pixel_pairwise_rms_coherent"
            ],
            (
                f"feature_diversity_coherent_beta_"
                f"{token}_class_{class_id}"
            ): row[
                "feature_cosine_diversity_coherent"
            ],
        }

        finite_metrics = {
            name: value
            for name, value in metrics.items()
            if math.isfinite(
                float(value)
            )
        }

        if finite_metrics:
            mlflow.log_metrics(
                finite_metrics
            )

    mlflow.log_metric(
        "duration_seconds",
        duration_seconds,
    )

    mlflow.log_artifacts(
        str(output_dir),
        artifact_path=(
            "generation_diversity"
        ),
    )


# ============================================================
# AFFICHAGE
# ============================================================

def print_model_results(
    summary: dict[str, Any],
    per_class_rows: list[
        dict[str, Any]
    ],
) -> None:
    """
    Afficher les résultats détaillés d'un CVAE.
    """

    print(
        "=" * 92
    )

    print(
        "CVAE : "
        f"{summary['checkpoint']}"
    )

    print(
        "=" * 92
    )

    print(
        "Beta                              : "
        f"{beta_label(summary['beta'])}"
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
        "-" * 92
    )

    print(
        f"{'ID':>2}  "
        f"{'Classe':<13} "
        f"{'Cohérence':>10}  "
        f"{'Pixel coh.':>11}  "
        f"{'Feature coh.':>13}"
    )

    print(
        "-" * 92
    )

    for row in per_class_rows:
        print(
            f"{row['class_id']:>2}  "
            f"{row['class_name']:<13} "
            f"{row['conditional_accuracy']:>9.2%}  "
            f"{row['pixel_pairwise_rms_coherent']:>11.6f}  "
            f"{row['feature_cosine_diversity_coherent']:>13.6f}"
        )

    print(
        "=" * 92
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    """
    Point d'entrée principal.
    """

    args = parse_args()

    start_time = time.perf_counter()

    set_random_seed(
        args.seed
    )

    device = select_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Classifieur indépendant
    # --------------------------------------------------------

    (
        classifier,
        classifier_validation_accuracy,
    ) = load_classifier(
        Path(
            args.classifier_checkpoint
        ),
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
            Path(path),
            device,
        )
        for path in args.checkpoints
    ]

    if not loaded_models:
        raise RuntimeError(
            "Aucun checkpoint CVAE à évaluer."
        )

    latent_dims = {
        loaded.latent_dim
        for loaded in loaded_models
    }

    if len(latent_dims) != 1:
        raise ValueError(
            "Les CVAE n'ont pas tous la même "
            "dimension latente : "
            f"{sorted(latent_dims)}"
        )

    latent_dim = (
        loaded_models[0].latent_dim
    )

    # --------------------------------------------------------
    # Banque z commune
    # --------------------------------------------------------

    latent_bank = create_shared_latent_bank(
        samples_per_class=(
            args.samples_per_class
        ),
        latent_dim=latent_dim,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # Informations générales
    # --------------------------------------------------------

    print(
        "=" * 92
    )

    print(
        "ÉVALUATION CONTRÔLÉE DE LA DIVERSITÉ "
        "DES GÉNÉRATIONS CVAE"
    )

    print(
        "=" * 92
    )

    print(
        "Appareil utilisé                  : "
        f"{device}"
    )

    print(
        "Classifieur                       : "
        f"{Path(args.classifier_checkpoint).name}"
    )

    if classifier_validation_accuracy is not None:
        print(
            "Accuracy validation classifieur   : "
            f"{classifier_validation_accuracy:.4%}"
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
        "Images par classe                 : "
        f"{args.samples_per_class}"
    )

    print(
        "Images générées par CVAE          : "
        f"{args.samples_per_class * len(FASHION_CLASSES)}"
    )

    print(
        "Seed des vecteurs latents         : "
        f"{args.seed}"
    )

    print(
        "Mêmes z entre les CVAE            : OUI"
    )

    print(
        "Mêmes z entre les classes         : OUI"
    )

    print(
        "Jeu officiel de test utilisé      : NON"
    )

    print(
        "=" * 92
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
                experiment_id=experiment_id,
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
            "=" * 92
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
                    "=" * 92
                )

            # ------------------------------------------------
            # Évaluation des CVAE
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
                    loaded=loaded,
                    classifier_features=(
                        classifier_features
                    ),
                    latent_bank=(
                        latent_bank
                    ),
                    batch_size=(
                        args.generation_batch_size
                    ),
                    device=device,
                )

                summary_rows.append(
                    summary
                )

                all_per_class_rows.extend(
                    per_class_rows
                )

                print_model_results(
                    summary,
                    per_class_rows,
                )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            summary_path = (
                output_dir
                / "generation_diversity_summary.csv"
            )

            per_class_path = (
                output_dir
                / "generation_diversity_per_class.csv"
            )

            write_csv(
                summary_path,
                summary_rows,
            )

            write_csv(
                per_class_path,
                all_per_class_rows,
            )

            # ------------------------------------------------
            # FIGURES
            # ------------------------------------------------

            feature_plot_path = (
                output_dir
                / "generation_diversity_feature_coherent.png"
            )

            pixel_plot_path = (
                output_dir
                / "generation_diversity_pixel_coherent.png"
            )

            per_class_feature_plot_path = (
                output_dir
                / "generation_diversity_feature_per_class.png"
            )

            save_summary_feature_plot(
                summary_rows,
                feature_plot_path,
            )

            save_summary_pixel_plot(
                summary_rows,
                pixel_plot_path,
            )

            save_per_class_feature_plot(
                all_per_class_rows,
                per_class_feature_plot_path,
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
                    mlflow=mlflow_module,
                    args=args,
                    device=device,
                    classifier_validation_accuracy=(
                        classifier_validation_accuracy
                    ),
                    summary_rows=(
                        summary_rows
                    ),
                    per_class_rows=(
                        all_per_class_rows
                    ),
                    output_dir=output_dir,
                    duration_seconds=(
                        duration_seconds
                    ),
                )

            # ------------------------------------------------
            # RÉSUMÉ FINAL
            # ------------------------------------------------

            print(
                "\n"
                + "=" * 92
            )

            print(
                "COMPARAISON FINALE DE LA DIVERSITÉ"
            )

            print(
                "=" * 92
            )

            print(
                f"{'Beta':>7}  "
                f"{'Cond. acc.':>11}  "
                f"{'Pixel coh.':>12}  "
                f"{'Feature coh.':>13}"
            )

            print(
                "-" * 92
            )

            for row in summary_rows:
                print(
                    f"{beta_label(row['beta']):>7}  "
                    f"{row['conditional_accuracy']:>10.2%}  "
                    f"{row['pixel_pairwise_rms_coherent_macro']:>12.6f}  "
                    f"{row['feature_cosine_diversity_coherent_macro']:>13.6f}"
                )

            print(
                "=" * 92
            )

            print(
                "Résumé CSV                         : "
                f"{summary_path}"
            )

            print(
                "Résultats par classe               : "
                f"{per_class_path}"
            )

            print(
                "Durée totale                       : "
                f"{duration_seconds:.2f} s"
            )

            print(
                "Jeu officiel de test               : "
                "NON UTILISÉ"
            )

            print(
                "=" * 92
            )

    finally:
        classifier_features.close()


if __name__ == "__main__":
    main()