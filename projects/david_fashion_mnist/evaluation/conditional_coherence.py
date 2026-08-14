"""
Évaluation multi-seed de la cohérence conditionnelle du CVAE Fashion-MNIST.

Objectif
--------
Mesurer quantitativement si les images générées par les CVAE finaux
(beta = 1) correspondent réellement à la classe demandée, puis mesurer
la robustesse de cette cohérence sur plusieurs seeds d'entraînement.

Le protocole utilise un classifieur Fashion-MNIST indépendant,
préalablement entraîné sur les vraies images du dataset.

Pour chaque checkpoint CVAE final :

1. on demande une classe Fashion-MNIST ;
2. on génère des images avec cette condition ;
3. le classifieur indépendant prédit la classe des images générées ;
4. on compare la classe prédite avec la classe demandée.

Comparaison contrôlée
---------------------
Une seule banque de vecteurs latents z ~ N(0, I) est créée et réutilisée :

- pour les dix classes ;
- pour les trois seeds d'entraînement ;
- pour tous les checkpoints évalués.

Ainsi, les différences entre seeds ne proviennent pas de tirages latents
différents.

Le jeu officiel de test Fashion-MNIST n'est jamais utilisé.

Sorties principales
-------------------
Le script sauvegarde :

- une matrice de cohérence brute par checkpoint ;
- une matrice normalisée par checkpoint ;
- une figure de matrice par checkpoint ;
- une grille qualitative contrôlée par checkpoint ;
- un CSV avec les résultats de chaque seed ;
- un CSV résumé moyenne ± écart-type entre seeds ;
- un CSV par classe et par seed ;
- un CSV résumé par classe moyenne ± écart-type ;
- deux figures de synthèse multi-seed.

Exemple final
-------------
    python -m evaluation.conditional_coherence \
        --device cuda \
        --checkpoints \
            checkpoints/cvae_beta_1_seed0_gpu_multiseed.pt \
            checkpoints/cvae_beta_1_seed42_final.pt \
            checkpoints/cvae_beta_1_seed123_gpu_multiseed.pt

Smoke test
----------
    python -m evaluation.conditional_coherence \
        --device cuda \
        --samples-per-class 20 \
        --grid-samples-per-class 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid

from evaluation.evaluate_classifier import (
    load_classifier,
    normalize_confusion_matrix,
    save_confusion_matrix_csv,
)
from models.cvae import CVAE
from models.fashion_classifier import (
    FASHION_MNIST_CLASSES,
    FashionMNISTClassifier,
)
from training.train_vae import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    configure_mlflow,
    select_device,
    set_random_seed,
)


# ================================================================
# CONFIGURATION PAR DÉFAUT
# ================================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "conditional_coherence_multiseed"
)

DEFAULT_CLASSIFIER_CHECKPOINT = (
    CHECKPOINT_DIR
    / "fashion_classifier_seed42_final.pt"
)

DEFAULT_CVAE_CHECKPOINTS = [
    CHECKPOINT_DIR
    / "cvae_beta_1_seed0_gpu_multiseed.pt",

    CHECKPOINT_DIR
    / "cvae_beta_1_seed42_final.pt",

    CHECKPOINT_DIR
    / "cvae_beta_1_seed123_gpu_multiseed.pt",
]

DEFAULT_EXPECTED_TRAINING_SEEDS = [
    0,
    42,
    123,
]

DEFAULT_MLFLOW_EXPERIMENT = (
    "fashion_mnist_final_multiseed_evaluation"
)

DEFAULT_MLFLOW_RUN_NAME = (
    "cvae_beta1_multiseed_conditional_coherence"
)


# ================================================================
# OUTILS GÉNÉRAUX
# ================================================================


def resolve_project_path(
    path: Path,
) -> Path:
    """
    Transforme un chemin relatif en chemin absolu du sous-projet.
    """

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def seed_to_tag(
    training_seed: int,
) -> str:
    """
    Transforme une seed en texte utilisable dans les métriques MLflow.
    """

    return str(
        training_seed
    ).replace(
        "-",
        "minus_",
    )


def sample_std(
    values: list[float],
) -> float:
    """
    Écart-type échantillonnal (ddof=1).

    Pour une seule valeur, retourne 0 afin de garder un CSV exploitable.
    Le protocole final utilise normalement trois seeds.
    """

    if len(values) <= 1:
        return 0.0

    return float(
        stdev(values)
    )


def extract_training_seed(
    checkpoint: dict,
) -> int:
    """
    Extrait la seed d'entraînement du checkpoint.

    Compatibilité :
    - checkpoints récents : configuration["training_seed"] ;
    - checkpoints historiques : configuration["seed"].
    """

    configuration = checkpoint.get(
        "configuration",
        {},
    )

    if (
        "training_seed"
        in configuration
    ):
        return int(
            configuration[
                "training_seed"
            ]
        )

    if "seed" in configuration:
        return int(
            configuration[
                "seed"
            ]
        )

    raise KeyError(
        "Impossible de déterminer la seed d'entraînement "
        "dans la configuration du checkpoint."
    )


def extract_checkpoint_split_seed(
    checkpoint: dict,
) -> Optional[int]:
    """
    Extrait split_seed lorsqu'il est explicitement présent.

    Les anciens checkpoints seed=42, créés avant le refactor,
    peuvent ne pas contenir ce champ.
    """

    configuration = checkpoint.get(
        "configuration",
        {},
    )

    if (
        "split_seed"
        not in configuration
    ):
        return None

    return int(
        configuration[
            "split_seed"
        ]
    )


# ================================================================
# CHARGEMENT D'UN CVAE
# ================================================================


def load_cvae(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CVAE, dict]:
    """
    Charge un checkpoint CVAE et reconstruit son architecture.
    """

    checkpoint_path = resolve_project_path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint CVAE introuvable : "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_type = checkpoint.get(
        "model_type"
    )

    if model_type != "CVAE":
        raise ValueError(
            "Le checkpoint ne correspond pas à un CVAE. "
            f"model_type reçu : {model_type!r}"
        )

    configuration = checkpoint.get(
        "configuration",
        {},
    )

    latent_dim = int(
        configuration.get(
            "latent_dim",
            16,
        )
    )

    hidden_dim = int(
        configuration.get(
            "hidden_dim",
            256,
        )
    )

    num_classes = int(
        configuration.get(
            "num_classes",
            10,
        )
    )

    if (
        num_classes
        != len(
            FASHION_MNIST_CLASSES
        )
    ):
        raise ValueError(
            "Le nombre de classes du CVAE est incompatible "
            "avec Fashion-MNIST. "
            f"Valeur reçue : {num_classes}."
        )

    model = CVAE(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# ================================================================
# BANQUE LATENTE CONTRÔLÉE
# ================================================================


def create_shared_latent_vectors(
    samples_per_class: int,
    latent_dim: int,
    seed: int,
) -> Tensor:
    """
    Crée une banque z unique, partagée entre classes et seeds.
    """

    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(seed)
    )

    return torch.randn(
        samples_per_class,
        latent_dim,
        generator=generator,
        device="cpu",
    )


# ================================================================
# MATRICE DE COHÉRENCE
# ================================================================


@torch.inference_mode()
def evaluate_conditional_coherence(
    cvae: CVAE,
    classifier: FashionMNISTClassifier,
    shared_latent_vectors: Tensor,
    device: torch.device,
    generation_batch_size: int,
) -> np.ndarray:
    """
    Mesure la cohérence conditionnelle d'un CVAE.

    Lignes :
        classe demandée au CVAE.

    Colonnes :
        classe prédite par le classifieur indépendant.
    """

    num_classes = len(
        FASHION_MNIST_CLASSES
    )

    samples_per_class = (
        shared_latent_vectors.shape[0]
    )

    confusion_matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    cvae.eval()
    classifier.eval()

    for requested_class in range(
        num_classes
    ):

        for start_index in range(
            0,
            samples_per_class,
            generation_batch_size,
        ):

            end_index = min(
                start_index
                + generation_batch_size,
                samples_per_class,
            )

            latent_batch = (
                shared_latent_vectors[
                    start_index:end_index
                ]
                .to(
                    device=device,
                    non_blocking=True,
                )
            )

            current_batch_size = (
                latent_batch.shape[0]
            )

            requested_labels = torch.full(
                size=(current_batch_size,),
                fill_value=requested_class,
                dtype=torch.long,
                device=device,
            )

            generated_images = cvae.decode(
                latent_batch,
                requested_labels,
            )

            classifier_logits = classifier(
                generated_images
            )

            predictions = torch.argmax(
                classifier_logits,
                dim=1,
            )

            predictions_cpu = (
                predictions
                .detach()
                .to("cpu")
            )

            predicted_counts = torch.bincount(
                predictions_cpu,
                minlength=num_classes,
            ).numpy()

            confusion_matrix[
                requested_class,
                :
            ] += predicted_counts

    return confusion_matrix


# ================================================================
# MÉTRIQUES DE COHÉRENCE
# ================================================================


def calculate_conditional_metrics(
    confusion_matrix: np.ndarray,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | str]],
]:
    """
    Calcule l'accuracy conditionnelle globale et par classe.
    """

    num_classes = len(
        FASHION_MNIST_CLASSES
    )

    expected_shape = (
        num_classes,
        num_classes,
    )

    if (
        confusion_matrix.shape
        != expected_shape
    ):
        raise ValueError(
            "La matrice doit avoir la forme "
            f"{expected_shape}. "
            f"Forme reçue : {confusion_matrix.shape}."
        )

    total_samples = int(
        confusion_matrix.sum()
    )

    correct_samples = int(
        np.trace(
            confusion_matrix
        )
    )

    if total_samples <= 0:
        raise RuntimeError(
            "Aucune image générée n'a été évaluée."
        )

    global_accuracy = (
        correct_samples
        / total_samples
    )

    global_metrics: dict[
        str,
        float | int
    ] = {
        "total_samples": (
            total_samples
        ),
        "correct_samples": (
            correct_samples
        ),
        "conditional_accuracy": (
            global_accuracy
        ),
    }

    per_class_rows: list[
        dict[
            str,
            float | int | str
        ]
    ] = []

    for class_index, class_name in enumerate(
        FASHION_MNIST_CLASSES
    ):

        requested_samples = int(
            confusion_matrix[
                class_index,
                :
            ].sum()
        )

        correct = int(
            confusion_matrix[
                class_index,
                class_index,
            ]
        )

        conditional_accuracy = (
            correct
            / requested_samples
            if requested_samples > 0
            else 0.0
        )

        per_class_rows.append(
            {
                "class_index": (
                    class_index
                ),
                "class_name": (
                    class_name
                ),
                "requested_samples": (
                    requested_samples
                ),
                "correct_predictions": (
                    correct
                ),
                "conditional_accuracy": (
                    conditional_accuracy
                ),
            }
        )

    return (
        global_metrics,
        per_class_rows,
    )


# ================================================================
# GRILLE QUALITATIVE CONTRÔLÉE
# ================================================================


@torch.inference_mode()
def save_controlled_generation_grid(
    cvae: CVAE,
    shared_latent_vectors: Tensor,
    samples_per_class: int,
    device: torch.device,
    output_path: Path,
    training_seed: int,
) -> None:
    """
    Sauvegarde une grille de générations contrôlées.

    Une colonne correspond au même z pour toutes les classes.
    """

    if samples_per_class <= 0:
        return

    samples_per_class = min(
        samples_per_class,
        shared_latent_vectors.shape[0],
    )

    controlled_z = (
        shared_latent_vectors[
            :samples_per_class
        ]
        .to(device)
    )

    generated_batches: list[
        Tensor
    ] = []

    for class_index in range(
        len(FASHION_MNIST_CLASSES)
    ):

        labels = torch.full(
            size=(samples_per_class,),
            fill_value=class_index,
            dtype=torch.long,
            device=device,
        )

        generated = cvae.decode(
            controlled_z,
            labels,
        )

        generated_batches.append(
            generated
            .detach()
            .cpu()
        )

    all_images = torch.cat(
        generated_batches,
        dim=0,
    )

    grid = make_grid(
        all_images,
        nrow=samples_per_class,
        padding=2,
        normalize=False,
        pad_value=1.0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            max(
                8,
                samples_per_class,
            ),
            10,
        )
    )

    axis.imshow(
        grid.permute(
            1,
            2,
            0,
        ).numpy()
    )

    axis.set_title(
        "Générations CVAE contrôlées "
        f"(training seed = {training_seed})\n"
        "Même vecteur latent z dans chaque colonne"
    )

    axis.axis(
        "off"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ================================================================
# FIGURE DE MATRICE
# ================================================================


def save_conditional_confusion_figure(
    normalized_confusion_matrix: np.ndarray,
    beta: float,
    training_seed: int,
    output_path: Path,
) -> None:
    """
    Sauvegarde la matrice classe demandée -> classe prédite.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    image = axis.imshow(
        normalized_confusion_matrix,
        vmin=0.0,
        vmax=1.0,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Proportion",
    )

    axis.set_title(
        "CVAE Fashion-MNIST - Cohérence conditionnelle\n"
        f"beta = {beta:g} | training seed = {training_seed}"
    )

    axis.set_xlabel(
        "Classe prédite par le classifieur"
    )

    axis.set_ylabel(
        "Classe demandée au CVAE"
    )

    indices = np.arange(
        len(FASHION_MNIST_CLASSES)
    )

    axis.set_xticks(
        indices
    )

    axis.set_yticks(
        indices
    )

    axis.set_xticklabels(
        FASHION_MNIST_CLASSES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        FASHION_MNIST_CLASSES
    )

    for requested_class in range(
        len(FASHION_MNIST_CLASSES)
    ):

        for predicted_class in range(
            len(FASHION_MNIST_CLASSES)
        ):

            value = (
                normalized_confusion_matrix[
                    requested_class,
                    predicted_class,
                ]
            )

            axis.text(
                predicted_class,
                requested_class,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ================================================================
# FIGURES MULTI-SEED
# ================================================================


def save_seed_accuracy_figure(
    rows: list[
        dict[
            str,
            float | int | str | None
        ]
    ],
    output_path: Path,
) -> None:
    """
    Barre de cohérence globale pour chaque seed.
    """

    ordered_rows = sorted(
        rows,
        key=lambda row: int(
            row[
                "training_seed"
            ]
        ),
    )

    seeds = [
        str(
            int(
                row[
                    "training_seed"
                ]
            )
        )
        for row in ordered_rows
    ]

    accuracies = [
        float(
            row[
                "conditional_accuracy"
            ]
        )
        for row in ordered_rows
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        seeds,
        accuracies,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "Training seed"
    )

    axis.set_ylabel(
        "Conditional accuracy"
    )

    axis.set_title(
        "CVAE beta=1 - Cohérence conditionnelle par seed"
    )

    for index, value in enumerate(
        accuracies
    ):
        axis.text(
            index,
            min(
                value + 0.02,
                0.98,
            ),
            f"{value:.2%}",
            ha="center",
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


def save_per_class_summary_figure(
    rows: list[
        dict[
            str,
            float | int
        ]
    ],
    output_path: Path,
) -> None:
    """
    Figure moyenne ± écart-type de cohérence par classe.
    """

    ordered_rows = sorted(
        rows,
        key=lambda row: int(
            row[
                "class_index"
            ]
        ),
    )

    labels = [
        str(
            row[
                "class_name"
            ]
        )
        for row in ordered_rows
    ]

    means = [
        float(
            row[
                "conditional_accuracy_mean"
            ]
        )
        for row in ordered_rows
    ]

    stds = [
        float(
            row[
                "conditional_accuracy_std"
            ]
        )
        for row in ordered_rows
    ]

    x = np.arange(
        len(labels)
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_ylabel(
        "Conditional accuracy"
    )

    axis.set_title(
        "CVAE beta=1 - Cohérence conditionnelle par classe\n"
        "Moyenne ± écart-type sur les seeds d'entraînement"
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


# ================================================================
# CSV
# ================================================================


def save_csv(
    rows: list[dict],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    """
    Sauvegarde une liste de dictionnaires dans un CSV.
    """

    if not rows:
        raise ValueError(
            "Aucune ligne disponible pour créer le CSV."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def build_global_multiseed_summary(
    rows: list[
        dict[
            str,
            float | int | str | None
        ]
    ],
    split_seed_protocol: int,
) -> list[dict]:
    """
    Agrège la cohérence globale sur les seeds d'entraînement.
    """

    accuracies = [
        float(
            row[
                "conditional_accuracy"
            ]
        )
        for row in rows
    ]

    total_generated = sum(
        int(
            row[
                "total_generated_images"
            ]
        )
        for row in rows
    )

    total_correct = sum(
        int(
            row[
                "correct_predictions"
            ]
        )
        for row in rows
    )

    seeds = sorted(
        int(
            row[
                "training_seed"
            ]
        )
        for row in rows
    )

    return [
        {
            "model_type": "CVAE",
            "beta": 1.0,
            "n_seeds": len(rows),
            "training_seeds": ",".join(
                str(seed)
                for seed in seeds
            ),
            "split_seed_protocol": (
                split_seed_protocol
            ),
            "total_generated_images_across_seeds": (
                total_generated
            ),
            "total_correct_predictions_across_seeds": (
                total_correct
            ),
            "conditional_accuracy_pooled": (
                total_correct
                / total_generated
            ),
            "conditional_accuracy_mean": (
                float(
                    mean(
                        accuracies
                    )
                )
            ),
            "conditional_accuracy_std": (
                sample_std(
                    accuracies
                )
            ),
            "conditional_accuracy_min": (
                min(
                    accuracies
                )
            ),
            "conditional_accuracy_max": (
                max(
                    accuracies
                )
            ),
        }
    ]


def build_per_class_multiseed_summary(
    rows: list[
        dict[
            str,
            float | int | str
        ]
    ],
) -> list[
    dict[
        str,
        float | int
    ]
]:
    """
    Agrège chaque classe sur les seeds d'entraînement.
    """

    summary_rows: list[
        dict[
            str,
            float | int
        ]
    ] = []

    for class_index, class_name in enumerate(
        FASHION_MNIST_CLASSES
    ):

        class_rows = [
            row
            for row in rows
            if int(
                row[
                    "class_index"
                ]
            )
            == class_index
        ]

        if not class_rows:
            continue

        accuracies = [
            float(
                row[
                    "conditional_accuracy"
                ]
            )
            for row in class_rows
        ]

        summary_rows.append(
            {
                "class_index": (
                    class_index
                ),
                "class_name": (
                    class_name
                ),
                "n_seeds": (
                    len(
                        accuracies
                    )
                ),
                "conditional_accuracy_mean": (
                    float(
                        mean(
                            accuracies
                        )
                    )
                ),
                "conditional_accuracy_std": (
                    sample_std(
                        accuracies
                    )
                ),
                "conditional_accuracy_min": (
                    min(
                        accuracies
                    )
                ),
                "conditional_accuracy_max": (
                    max(
                        accuracies
                    )
                ),
            }
        )

    return summary_rows


# ================================================================
# AFFICHAGE D'UN MODÈLE
# ================================================================


def print_model_results(
    checkpoint_name: str,
    beta: float,
    training_seed: int,
    epoch: int,
    global_metrics: dict[
        str,
        float | int
    ],
    per_class_metrics: list[
        dict[
            str,
            float | int | str
        ]
    ],
) -> None:
    """
    Affiche les résultats d'un checkpoint CVAE.
    """

    print("=" * 96)

    print(
        f"CVAE : {checkpoint_name}"
    )

    print("=" * 96)

    print(
        f"Beta                       : "
        f"{beta:g}"
    )

    print(
        f"Training seed              : "
        f"{training_seed}"
    )

    print(
        f"Époque du checkpoint       : "
        f"{epoch}"
    )

    print(
        f"Images générées            : "
        f"{int(global_metrics['total_samples'])}"
    )

    print(
        f"Prédictions correctes      : "
        f"{int(global_metrics['correct_samples'])}"
    )

    print(
        f"Conditional accuracy       : "
        f"{float(global_metrics['conditional_accuracy']):.4%}"
    )

    print("-" * 96)

    print(
        f"{'ID':>2}  "
        f"{'Classe demandée':<14} "
        f"{'Images':>8} "
        f"{'Correct':>8} "
        f"{'Cohérence':>11}"
    )

    print("-" * 96)

    for row in per_class_metrics:

        print(
            f"{int(row['class_index']):>2}  "
            f"{str(row['class_name']):<14} "
            f"{int(row['requested_samples']):>8} "
            f"{int(row['correct_predictions']):>8} "
            f"{float(row['conditional_accuracy']):>10.2%}"
        )

    print("=" * 96)


# ================================================================
# ARGUMENTS
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments disponibles.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Mesurer la cohérence conditionnelle multi-seed "
            "du CVAE Fashion-MNIST."
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
            "Checkpoint du classifieur indépendant."
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
        default=DEFAULT_EXPECTED_TRAINING_SEEDS,
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
            "Elle est auditée dans les checkpoints récents. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help=(
            "Nombre d'images générées pour chaque classe "
            "et chaque seed. Valeur par défaut : 1000."
        ),
    )

    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=256,
        help=(
            "Nombre d'images générées simultanément. "
            "Valeur par défaut : 256."
        ),
    )

    parser.add_argument(
        "--grid-samples-per-class",
        type=int,
        default=10,
        help=(
            "Nombre d'images par classe dans les grilles "
            "qualitatives contrôlées. Utiliser 0 pour désactiver."
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
        default=RESULTS_DIR,
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

    return parser


# ================================================================
# VALIDATION DES ARGUMENTS
# ================================================================


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Vérifie les arguments reçus.
    """

    if args.samples_per_class <= 0:
        raise ValueError(
            "samples-per-class doit être strictement positif."
        )

    if args.generation_batch_size <= 0:
        raise ValueError(
            "generation-batch-size doit être strictement positif."
        )

    if args.grid_samples_per_class < 0:
        raise ValueError(
            "grid-samples-per-class ne peut pas être négatif."
        )

    if len(args.checkpoints) == 0:
        raise ValueError(
            "Au moins un checkpoint CVAE doit être fourni."
        )

    if len(args.expected_training_seeds) == 0:
        raise ValueError(
            "Au moins une seed d'entraînement attendue doit être fournie."
        )

    if (
        len(
            set(
                args.expected_training_seeds
            )
        )
        != len(
            args.expected_training_seeds
        )
    ):
        raise ValueError(
            "expected-training-seeds contient des doublons."
        )

    if not args.mlflow_experiment_name.strip():
        raise ValueError(
            "mlflow-experiment-name ne peut pas être vide."
        )

    if not args.mlflow_run_name.strip():
        raise ValueError(
            "mlflow-run-name ne peut pas être vide."
        )


# ================================================================
# PROGRAMME PRINCIPAL
# ================================================================


def main() -> None:
    """
    Point d'entrée principal.
    """

    parser = build_argument_parser()

    args = parser.parse_args()

    validate_arguments(
        args
    )

    set_random_seed(
        args.seed
    )

    device = select_device(
        args.device
    )

    output_dir = resolve_project_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier_checkpoint_path = resolve_project_path(
        args.classifier_checkpoint
    )

    classifier, classifier_checkpoint = load_classifier(
        checkpoint_path=classifier_checkpoint_path,
        device=device,
    )

    classifier_validation_accuracy = (
        classifier_checkpoint.get(
            "validation_accuracy"
        )
    )

    if (
        classifier_validation_accuracy
        is None
    ):
        raise KeyError(
            "Le checkpoint du classifieur ne contient pas "
            "validation_accuracy."
        )

    # ------------------------------------------------------------
    # Charger et auditer tous les CVAE avant toute génération.
    # ------------------------------------------------------------

    loaded_models: list[
        tuple[
            Path,
            CVAE,
            dict,
            int,
            Optional[int],
        ]
    ] = []

    latent_dimensions: set[int] = set()
    observed_training_seeds: list[int] = []

    for checkpoint_argument in args.checkpoints:

        checkpoint_path = resolve_project_path(
            checkpoint_argument
        )

        cvae, checkpoint = load_cvae(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        beta = float(
            checkpoint.get(
                "beta",
                checkpoint.get(
                    "configuration",
                    {},
                ).get(
                    "beta",
                    float("nan"),
                ),
            )
        )

        if not np.isclose(
            beta,
            args.expected_beta,
        ):
            raise ValueError(
                f"{checkpoint_path.name} : beta={beta} "
                f"mais beta attendu={args.expected_beta}."
            )

        training_seed = extract_training_seed(
            checkpoint
        )

        checkpoint_split_seed = (
            extract_checkpoint_split_seed(
                checkpoint
            )
        )

        if (
            checkpoint_split_seed
            is not None
            and checkpoint_split_seed
            != args.split_seed_protocol
        ):
            raise ValueError(
                f"{checkpoint_path.name} : split_seed="
                f"{checkpoint_split_seed}, attendu="
                f"{args.split_seed_protocol}."
            )

        latent_dimensions.add(
            cvae.latent_dim
        )

        observed_training_seeds.append(
            training_seed
        )

        loaded_models.append(
            (
                checkpoint_path,
                cvae,
                checkpoint,
                training_seed,
                checkpoint_split_seed,
            )
        )

    if len(latent_dimensions) != 1:
        raise ValueError(
            "Tous les CVAE doivent avoir la même dimension latente. "
            f"Dimensions observées : {sorted(latent_dimensions)}"
        )

    if (
        len(
            set(
                observed_training_seeds
            )
        )
        != len(
            observed_training_seeds
        )
    ):
        raise ValueError(
            "Plusieurs checkpoints possèdent la même seed d'entraînement : "
            f"{observed_training_seeds}"
        )

    expected_seed_set = set(
        args.expected_training_seeds
    )

    observed_seed_set = set(
        observed_training_seeds
    )

    if (
        observed_seed_set
        != expected_seed_set
    ):
        raise ValueError(
            "Les seeds des checkpoints ne correspondent pas "
            "au protocole attendu. "
            f"Observées={sorted(observed_seed_set)}, "
            f"attendues={sorted(expected_seed_set)}."
        )

    latent_dim = next(
        iter(
            latent_dimensions
        )
    )

    shared_latent_vectors = create_shared_latent_vectors(
        samples_per_class=args.samples_per_class,
        latent_dim=latent_dim,
        seed=args.seed,
    )

    total_images_per_model = (
        args.samples_per_class
        * len(
            FASHION_MNIST_CLASSES
        )
    )

    print("=" * 96)
    print(
        "ÉVALUATION MULTI-SEED DE LA COHÉRENCE "
        "CONDITIONNELLE DU CVAE"
    )
    print("=" * 96)
    print(
        f"Appareil utilisé                  : "
        f"{device}"
    )
    print(
        f"Classifieur                       : "
        f"{classifier_checkpoint_path.name}"
    )
    print(
        f"Accuracy validation classifieur   : "
        f"{float(classifier_validation_accuracy):.4%}"
    )
    print(
        f"Beta attendu                      : "
        f"{args.expected_beta:g}"
    )
    print(
        f"Training seeds                    : "
        f"{sorted(observed_training_seeds)}"
    )
    print(
        f"Split seed du protocole           : "
        f"{args.split_seed_protocol}"
    )
    print(
        f"Dimension latente                 : "
        f"{latent_dim}"
    )
    print(
        f"Images par classe et par seed     : "
        f"{args.samples_per_class}"
    )
    print(
        f"Images générées par checkpoint    : "
        f"{total_images_per_model}"
    )
    print(
        f"Seed des vecteurs latents         : "
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
    print("=" * 96)

    summary_rows: list[
        dict[
            str,
            float | int | str | None
        ]
    ] = []

    all_per_class_rows: list[
        dict[
            str,
            float | int | str
        ]
    ] = []

    # ------------------------------------------------------------
    # Configuration MLflow
    # ------------------------------------------------------------

    tracking_uri: Optional[str] = None

    if not args.disable_mlflow:

        tracking_uri = configure_mlflow(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment_name,
        )

    # ------------------------------------------------------------
    # Évaluation des checkpoints.
    # ------------------------------------------------------------

    def run_evaluations() -> None:
        """
        Évalue tous les checkpoints avec la même banque latente.
        """

        for (
            checkpoint_path,
            cvae,
            checkpoint,
            training_seed,
            checkpoint_split_seed,
        ) in sorted(
            loaded_models,
            key=lambda item: item[3],
        ):

            beta = float(
                checkpoint.get(
                    "beta",
                    args.expected_beta,
                )
            )

            epoch = int(
                checkpoint.get(
                    "epoch",
                    0,
                )
            )

            print(
                f"\nÉvaluation de "
                f"{checkpoint_path.name}..."
            )

            confusion_matrix = evaluate_conditional_coherence(
                cvae=cvae,
                classifier=classifier,
                shared_latent_vectors=shared_latent_vectors,
                device=device,
                generation_batch_size=args.generation_batch_size,
            )

            (
                global_metrics,
                per_class_metrics,
            ) = calculate_conditional_metrics(
                confusion_matrix
            )

            normalized_confusion = normalize_confusion_matrix(
                confusion_matrix
            )

            raw_matrix_path = (
                output_dir
                / (
                    f"{checkpoint_path.stem}"
                    "_conditional_confusion_matrix.csv"
                )
            )

            normalized_matrix_path = (
                output_dir
                / (
                    f"{checkpoint_path.stem}"
                    "_conditional_confusion_matrix_normalized.csv"
                )
            )

            matrix_figure_path = (
                output_dir
                / (
                    f"{checkpoint_path.stem}"
                    "_conditional_confusion_matrix.png"
                )
            )

            generation_grid_path = (
                output_dir
                / (
                    f"{checkpoint_path.stem}"
                    "_controlled_generations.png"
                )
            )

            save_confusion_matrix_csv(
                confusion_matrix=confusion_matrix,
                output_path=raw_matrix_path,
            )

            save_confusion_matrix_csv(
                confusion_matrix=normalized_confusion,
                output_path=normalized_matrix_path,
                decimal_places=6,
            )

            save_conditional_confusion_figure(
                normalized_confusion_matrix=normalized_confusion,
                beta=beta,
                training_seed=training_seed,
                output_path=matrix_figure_path,
            )

            if (
                args.grid_samples_per_class
                > 0
            ):

                save_controlled_generation_grid(
                    cvae=cvae,
                    shared_latent_vectors=shared_latent_vectors,
                    samples_per_class=args.grid_samples_per_class,
                    device=device,
                    output_path=generation_grid_path,
                    training_seed=training_seed,
                )

            summary_rows.append(
                {
                    "checkpoint": (
                        checkpoint_path.name
                    ),
                    "model_type": "CVAE",
                    "beta": beta,
                    "training_seed": (
                        training_seed
                    ),
                    "split_seed_protocol": (
                        args.split_seed_protocol
                    ),
                    "checkpoint_split_seed": (
                        checkpoint_split_seed
                    ),
                    "best_epoch": epoch,
                    "samples_per_class": (
                        args.samples_per_class
                    ),
                    "total_generated_images": int(
                        global_metrics[
                            "total_samples"
                        ]
                    ),
                    "correct_predictions": int(
                        global_metrics[
                            "correct_samples"
                        ]
                    ),
                    "conditional_accuracy": float(
                        global_metrics[
                            "conditional_accuracy"
                        ]
                    ),
                    "latent_seed": (
                        args.seed
                    ),
                    "same_latents_across_seeds": (
                        True
                    ),
                    "same_latents_across_classes": (
                        True
                    ),
                    "classifier_checkpoint": (
                        classifier_checkpoint_path.name
                    ),
                    "classifier_validation_accuracy": (
                        float(
                            classifier_validation_accuracy
                        )
                    ),
                }
            )

            for class_row in per_class_metrics:

                all_per_class_rows.append(
                    {
                        "checkpoint": (
                            checkpoint_path.name
                        ),
                        "beta": beta,
                        "training_seed": (
                            training_seed
                        ),
                        "class_index": (
                            class_row[
                                "class_index"
                            ]
                        ),
                        "class_name": (
                            class_row[
                                "class_name"
                            ]
                        ),
                        "requested_samples": (
                            class_row[
                                "requested_samples"
                            ]
                        ),
                        "correct_predictions": (
                            class_row[
                                "correct_predictions"
                            ]
                        ),
                        "conditional_accuracy": (
                            class_row[
                                "conditional_accuracy"
                            ]
                        ),
                        "latent_seed": (
                            args.seed
                        ),
                    }
                )

            print_model_results(
                checkpoint_name=checkpoint_path.name,
                beta=beta,
                training_seed=training_seed,
                epoch=epoch,
                global_metrics=global_metrics,
                per_class_metrics=per_class_metrics,
            )

            if not args.disable_mlflow:

                seed_tag = seed_to_tag(
                    training_seed
                )

                mlflow.log_metric(
                    f"conditional_accuracy_seed_{seed_tag}",
                    float(
                        global_metrics[
                            "conditional_accuracy"
                        ]
                    ),
                )

                for class_row in per_class_metrics:

                    class_index = int(
                        class_row[
                            "class_index"
                        ]
                    )

                    mlflow.log_metric(
                        (
                            "conditional_accuracy_"
                            f"seed_{seed_tag}_"
                            f"class_{class_index}"
                        ),
                        float(
                            class_row[
                                "conditional_accuracy"
                            ]
                        ),
                    )

                artifact_path = (
                    f"seed_{seed_tag}"
                )

                mlflow.log_artifact(
                    str(
                        raw_matrix_path
                    ),
                    artifact_path=artifact_path,
                )

                mlflow.log_artifact(
                    str(
                        normalized_matrix_path
                    ),
                    artifact_path=artifact_path,
                )

                mlflow.log_artifact(
                    str(
                        matrix_figure_path
                    ),
                    artifact_path=artifact_path,
                )

                if (
                    args.grid_samples_per_class
                    > 0
                    and generation_grid_path.exists()
                ):

                    mlflow.log_artifact(
                        str(
                            generation_grid_path
                        ),
                        artifact_path=artifact_path,
                    )

    # ------------------------------------------------------------
    # Exécuter avec ou sans MLflow.
    # ------------------------------------------------------------

    active_run_id: Optional[str] = None

    if args.disable_mlflow:

        run_evaluations()

    else:

        with mlflow.start_run(
            run_name=args.mlflow_run_name
        ) as active_run:

            active_run_id = (
                active_run.info.run_id
            )

            print(
                f"MLflow run ID                     : "
                f"{active_run_id}"
            )

            print(
                f"MLflow tracking URI               : "
                f"{tracking_uri}"
            )

            print("=" * 96)

            mlflow.log_params(
                {
                    "evaluation": (
                        "conditional_coherence_multiseed"
                    ),
                    "dataset": (
                        "Fashion-MNIST"
                    ),
                    "model_type": (
                        "CVAE"
                    ),
                    "beta": (
                        args.expected_beta
                    ),
                    "training_seeds": (
                        ",".join(
                            str(seed)
                            for seed in sorted(
                                observed_training_seeds
                            )
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
                    "latent_dim": (
                        latent_dim
                    ),
                    "latent_seed": (
                        args.seed
                    ),
                    "num_classes": (
                        len(
                            FASHION_MNIST_CLASSES
                        )
                    ),
                    "same_latents_across_seeds": (
                        True
                    ),
                    "same_latents_across_classes": (
                        True
                    ),
                    "classifier_checkpoint": (
                        classifier_checkpoint_path.name
                    ),
                    "classifier_validation_accuracy": (
                        float(
                            classifier_validation_accuracy
                        )
                    ),
                    "test_used": (
                        False
                    ),
                }
            )

            mlflow.set_tags(
                {
                    "dataset": (
                        "Fashion-MNIST"
                    ),
                    "evaluation_type": (
                        "conditional_coherence_multiseed"
                    ),
                    "evaluator": (
                        "independent_classifier"
                    ),
                    "comparison_control": (
                        "shared_latent_vectors"
                    ),
                    "test_used": (
                        "False"
                    ),
                }
            )

            run_evaluations()

    # ------------------------------------------------------------
    # Construire les résumés multi-seed.
    # ------------------------------------------------------------

    global_summary_rows = build_global_multiseed_summary(
        rows=summary_rows,
        split_seed_protocol=args.split_seed_protocol,
    )

    per_class_summary_rows = build_per_class_multiseed_summary(
        rows=all_per_class_rows
    )

    runs_path = (
        output_dir
        / "conditional_coherence_multiseed_runs.csv"
    )

    global_summary_path = (
        output_dir
        / "conditional_coherence_multiseed_summary.csv"
    )

    per_class_path = (
        output_dir
        / "conditional_coherence_multiseed_per_class.csv"
    )

    per_class_summary_path = (
        output_dir
        / "conditional_coherence_multiseed_per_class_summary.csv"
    )

    seed_figure_path = (
        output_dir
        / "conditional_coherence_multiseed_by_seed.png"
    )

    per_class_figure_path = (
        output_dir
        / "conditional_coherence_multiseed_per_class_mean_std.png"
    )

    save_csv(
        rows=summary_rows,
        output_path=runs_path,
        fieldnames=[
            "checkpoint",
            "model_type",
            "beta",
            "training_seed",
            "split_seed_protocol",
            "checkpoint_split_seed",
            "best_epoch",
            "samples_per_class",
            "total_generated_images",
            "correct_predictions",
            "conditional_accuracy",
            "latent_seed",
            "same_latents_across_seeds",
            "same_latents_across_classes",
            "classifier_checkpoint",
            "classifier_validation_accuracy",
        ],
    )

    save_csv(
        rows=global_summary_rows,
        output_path=global_summary_path,
        fieldnames=[
            "model_type",
            "beta",
            "n_seeds",
            "training_seeds",
            "split_seed_protocol",
            "total_generated_images_across_seeds",
            "total_correct_predictions_across_seeds",
            "conditional_accuracy_pooled",
            "conditional_accuracy_mean",
            "conditional_accuracy_std",
            "conditional_accuracy_min",
            "conditional_accuracy_max",
        ],
    )

    save_csv(
        rows=all_per_class_rows,
        output_path=per_class_path,
        fieldnames=[
            "checkpoint",
            "beta",
            "training_seed",
            "class_index",
            "class_name",
            "requested_samples",
            "correct_predictions",
            "conditional_accuracy",
            "latent_seed",
        ],
    )

    save_csv(
        rows=per_class_summary_rows,
        output_path=per_class_summary_path,
        fieldnames=[
            "class_index",
            "class_name",
            "n_seeds",
            "conditional_accuracy_mean",
            "conditional_accuracy_std",
            "conditional_accuracy_min",
            "conditional_accuracy_max",
        ],
    )

    save_seed_accuracy_figure(
        rows=summary_rows,
        output_path=seed_figure_path,
    )

    save_per_class_summary_figure(
        rows=per_class_summary_rows,
        output_path=per_class_figure_path,
    )

    # ------------------------------------------------------------
    # Log des résumés MLflow.
    # ------------------------------------------------------------

    if not args.disable_mlflow:

        global_summary = (
            global_summary_rows[0]
        )

        mlflow_client = mlflow.tracking.MlflowClient()

        if active_run_id is None:
            raise RuntimeError(
                "Run ID MLflow absent après l'évaluation."
            )

        mlflow_client.log_metric(
            active_run_id,
            "conditional_accuracy_mean",
            float(
                global_summary[
                    "conditional_accuracy_mean"
                ]
            ),
        )

        mlflow_client.log_metric(
            active_run_id,
            "conditional_accuracy_std",
            float(
                global_summary[
                    "conditional_accuracy_std"
                ]
            ),
        )

        mlflow_client.log_metric(
            active_run_id,
            "conditional_accuracy_pooled",
            float(
                global_summary[
                    "conditional_accuracy_pooled"
                ]
            ),
        )

        for class_row in per_class_summary_rows:

            class_index = int(
                class_row[
                    "class_index"
                ]
            )

            mlflow_client.log_metric(
                active_run_id,
                (
                    "conditional_accuracy_"
                    f"class_{class_index}_mean"
                ),
                float(
                    class_row[
                        "conditional_accuracy_mean"
                    ]
                ),
            )

            mlflow_client.log_metric(
                active_run_id,
                (
                    "conditional_accuracy_"
                    f"class_{class_index}_std"
                ),
                float(
                    class_row[
                        "conditional_accuracy_std"
                    ]
                ),
            )

        for artifact in [
            runs_path,
            global_summary_path,
            per_class_path,
            per_class_summary_path,
            seed_figure_path,
            per_class_figure_path,
        ]:

            mlflow_client.log_artifact(
                active_run_id,
                str(
                    artifact
                ),
                artifact_path="comparison",
            )

    # ------------------------------------------------------------
    # Résumé final.
    # ------------------------------------------------------------

    print("\n" + "=" * 96)
    print(
        "COHÉRENCE CONDITIONNELLE MULTI-SEED - RÉSUMÉ FINAL"
    )
    print("=" * 96)

    print(
        f"{'Seed':>8} "
        f"{'Images':>10} "
        f"{'Correctes':>12} "
        f"{'Conditional accuracy':>22}"
    )

    print("-" * 96)

    for row in sorted(
        summary_rows,
        key=lambda item: int(
            item[
                "training_seed"
            ]
        ),
    ):

        print(
            f"{int(row['training_seed']):>8} "
            f"{int(row['total_generated_images']):>10} "
            f"{int(row['correct_predictions']):>12} "
            f"{float(row['conditional_accuracy']):>21.2%}"
        )

    global_summary = (
        global_summary_rows[0]
    )

    print("-" * 96)

    print(
        "Moyenne ± écart-type              : "
        f"{float(global_summary['conditional_accuracy_mean']):.2%} "
        "± "
        f"{float(global_summary['conditional_accuracy_std']):.2%}"
    )

    print(
        "Accuracy pooled                   : "
        f"{float(global_summary['conditional_accuracy_pooled']):.2%}"
    )

    print("=" * 96)

    print(
        f"Runs CSV                          : "
        f"{runs_path}"
    )

    print(
        f"Résumé multi-seed                 : "
        f"{global_summary_path}"
    )

    print(
        f"Résultats par classe              : "
        f"{per_class_path}"
    )

    print(
        f"Résumé par classe                 : "
        f"{per_class_summary_path}"
    )

    print(
        f"Figure par seed                   : "
        f"{seed_figure_path}"
    )

    print(
        f"Figure par classe                 : "
        f"{per_class_figure_path}"
    )

    print(
        "Jeu officiel de test              : NON UTILISÉ"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()