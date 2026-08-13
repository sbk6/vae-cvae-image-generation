"""
Évaluation du classifieur Fashion-MNIST sur le jeu de validation.

Ce script évalue le classifieur indépendant utilisé ensuite comme
évaluateur externe des images générées par le CVAE.

Important
---------
Le jeu officiel de test Fashion-MNIST n'est PAS utilisé ici.

On réutilise exactement le protocole train / validation du projet :

- 54 000 images d'entraînement ;
- 6 000 images de validation ;
- split déterministe avec seed 42 par défaut.

L'objectif est de mesurer la fiabilité du classifieur sur de vraies
images Fashion-MNIST avant de lui demander d'évaluer les images
générées par les CVAE.

Métriques calculées
-------------------
- accuracy globale ;
- accuracy par classe ;
- précision par classe ;
- rappel par classe ;
- F1-score par classe ;
- support de chaque classe ;
- matrice de confusion brute ;
- matrice de confusion normalisée.

Exemple
-------
    python -m evaluation.evaluate_classifier

Avec CUDA :
    python -m evaluation.evaluate_classifier --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.fashion_classifier import (
    FASHION_MNIST_CLASSES,
    FashionMNISTClassifier,
)
from training.train_vae import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    create_dataloaders,
    select_device,
    set_random_seed,
)


# ================================================================
# CHEMINS
# ================================================================

RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_CHECKPOINT = (
    CHECKPOINT_DIR
    / "fashion_classifier_seed42_final.pt"
)

DEFAULT_METRICS_CSV = (
    RESULTS_DIR
    / "classifier_validation_metrics.csv"
)

DEFAULT_CONFUSION_CSV = (
    RESULTS_DIR
    / "classifier_validation_confusion_matrix.csv"
)

DEFAULT_CONFUSION_NORMALIZED_CSV = (
    RESULTS_DIR
    / "classifier_validation_confusion_matrix_normalized.csv"
)

DEFAULT_CONFUSION_FIGURE = (
    RESULTS_DIR
    / "classifier_validation_confusion_matrix.png"
)


# ================================================================
# CHARGEMENT DU CHECKPOINT
# ================================================================


def load_classifier(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FashionMNISTClassifier, dict]:
    """
    Charge le classifieur depuis un checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Chemin du checkpoint PyTorch.

    device:
        CPU ou CUDA.

    Returns
    -------
    model:
        Classifieur chargé et placé en mode évaluation.

    checkpoint:
        Dictionnaire complet du checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint introuvable : "
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

    if model_type != "FashionMNISTClassifier":
        raise ValueError(
            "Le checkpoint ne correspond pas au "
            "FashionMNISTClassifier. "
            f"model_type reçu : {model_type!r}"
        )

    configuration = checkpoint.get(
        "configuration",
        {},
    )

    num_classes = configuration.get(
        "num_classes",
        10,
    )

    if num_classes != len(FASHION_MNIST_CLASSES):
        raise ValueError(
            "Nombre de classes incompatible avec "
            "Fashion-MNIST. "
            f"Valeur reçue : {num_classes}."
        )

    model = FashionMNISTClassifier(
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ================================================================
# MATRICE DE CONFUSION
# ================================================================


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> None:
    """
    Met à jour une matrice de confusion.

    Convention
    ----------
    Les lignes représentent les vraies classes.

    Les colonnes représentent les classes prédites.

    Exemple
    -------
    confusion_matrix[3, 6] correspond au nombre
    d'images réellement de classe 3 mais prédites
    comme classe 6.
    """

    if targets.shape != predictions.shape:
        raise ValueError(
            "targets et predictions doivent avoir "
            "exactement la même forme."
        )

    targets_cpu = (
        targets
        .detach()
        .to("cpu")
        .long()
    )

    predictions_cpu = (
        predictions
        .detach()
        .to("cpu")
        .long()
    )

    combined_indices = (
        targets_cpu * num_classes
        + predictions_cpu
    )

    batch_counts = torch.bincount(
        combined_indices,
        minlength=num_classes * num_classes,
    )

    batch_matrix = batch_counts.reshape(
        num_classes,
        num_classes,
    )

    confusion_matrix += batch_matrix


# ================================================================
# ÉVALUATION
# ================================================================


@torch.no_grad()
def evaluate_classifier(
    model: FashionMNISTClassifier,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Évalue le classifieur sur un DataLoader.

    Returns
    -------
    global_metrics:
        Accuracy globale et nombre d'images traitées.

    confusion_matrix:
        Matrice de confusion brute.
    """

    model.eval()

    num_classes = len(
        FASHION_MNIST_CLASSES
    )

    confusion_matrix = torch.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=torch.long,
    )

    processed_samples = 0
    correct_predictions = 0

    for batch_index, (
        images,
        labels,
    ) in enumerate(dataloader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device=device,
            non_blocking=True,
        )

        labels = labels.to(
            device=device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct_predictions += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        processed_samples += (
            labels.shape[0]
        )

        update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            targets=labels,
            predictions=predictions,
            num_classes=num_classes,
        )

    if processed_samples <= 0:
        raise RuntimeError(
            "Aucune image n'a été évaluée."
        )

    overall_accuracy = (
        correct_predictions
        / processed_samples
    )

    global_metrics = {
        "processed_samples": float(
            processed_samples
        ),
        "correct_predictions": float(
            correct_predictions
        ),
        "overall_accuracy": (
            overall_accuracy
        ),
    }

    return (
        global_metrics,
        confusion_matrix.numpy(),
    )


# ================================================================
# MÉTRIQUES PAR CLASSE
# ================================================================


def calculate_per_class_metrics(
    confusion_matrix: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """
    Calcule les métriques de chaque classe.

    Pour une classe donnée :

    support
        Nombre réel d'images de cette classe.

    accuracy_class
        Nombre correctement reconnus dans cette classe
        divisé par le nombre réel d'images de cette classe.

        Cette quantité est mathématiquement identique au rappel
        de cette classe.

    precision
        Parmi toutes les images prédites comme appartenant à
        cette classe, proportion réellement correcte.

    recall
        Parmi toutes les vraies images de cette classe,
        proportion correctement reconnue.

    f1_score
        Moyenne harmonique de précision et rappel.
    """

    num_classes = len(
        FASHION_MNIST_CLASSES
    )

    if confusion_matrix.shape != (
        num_classes,
        num_classes,
    ):
        raise ValueError(
            "La matrice de confusion doit avoir la forme "
            f"({num_classes}, {num_classes}). "
            f"Forme reçue : {confusion_matrix.shape}."
        )

    total_samples = int(
        confusion_matrix.sum()
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    for class_index, class_name in enumerate(
        FASHION_MNIST_CLASSES
    ):

        true_positive = int(
            confusion_matrix[
                class_index,
                class_index,
            ]
        )

        support = int(
            confusion_matrix[
                class_index,
                :
            ].sum()
        )

        predicted_as_class = int(
            confusion_matrix[
                :,
                class_index,
            ].sum()
        )

        false_positive = (
            predicted_as_class
            - true_positive
        )

        false_negative = (
            support
            - true_positive
        )

        true_negative = (
            total_samples
            - true_positive
            - false_positive
            - false_negative
        )

        class_accuracy = (
            true_positive / support
            if support > 0
            else 0.0
        )

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
            if (
                true_positive
                + false_positive
            ) > 0
            else 0.0
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
            else 0.0
        )

        if (
            precision
            + recall
        ) > 0:

            f1_score = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1_score = 0.0

        rows.append(
            {
                "class_index": (
                    class_index
                ),
                "class_name": (
                    class_name
                ),
                "support": (
                    support
                ),
                "correct": (
                    true_positive
                ),
                "predicted_as_class": (
                    predicted_as_class
                ),
                "true_positive": (
                    true_positive
                ),
                "false_positive": (
                    false_positive
                ),
                "false_negative": (
                    false_negative
                ),
                "true_negative": (
                    true_negative
                ),
                "class_accuracy": (
                    class_accuracy
                ),
                "precision": (
                    precision
                ),
                "recall": (
                    recall
                ),
                "f1_score": (
                    f1_score
                ),
            }
        )

    return rows


# ================================================================
# NORMALISATION DE LA MATRICE DE CONFUSION
# ================================================================


def normalize_confusion_matrix(
    confusion_matrix: np.ndarray,
) -> np.ndarray:
    """
    Normalise chaque ligne de la matrice de confusion.

    Après normalisation, chaque ligne représente une distribution
    de probabilités conditionnelle à la vraie classe.

    Une ligne doit donc sommer approximativement à 1.
    """

    row_sums = confusion_matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        confusion_matrix.astype(
            np.float64
        ),
        row_sums,
        out=np.zeros_like(
            confusion_matrix,
            dtype=np.float64,
        ),
        where=(
            row_sums != 0
        ),
    )

    return normalized


# ================================================================
# SAUVEGARDE DES MÉTRIQUES
# ================================================================


def save_metrics_csv(
    rows: list[
        dict[str, float | int | str]
    ],
    global_metrics: dict[str, float],
    checkpoint: dict,
    output_path: Path,
) -> None:
    """
    Sauvegarde les métriques globales et par classe.

    La première ligne du CSV représente le résultat GLOBAL.
    Les lignes suivantes correspondent aux dix classes.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "scope",
        "class_index",
        "class_name",
        "support",
        "correct",
        "predicted_as_class",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "class_accuracy",
        "precision",
        "recall",
        "f1_score",
        "overall_accuracy",
        "checkpoint_epoch",
        "checkpoint_validation_loss",
        "checkpoint_validation_accuracy",
    ]

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    checkpoint_validation_loss = checkpoint.get(
        "best_validation_loss"
    )

    checkpoint_validation_accuracy = checkpoint.get(
        "validation_accuracy"
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

        # Ligne globale.
        writer.writerow(
            {
                "scope": "GLOBAL",
                "class_index": "",
                "class_name": "ALL_CLASSES",
                "support": int(
                    global_metrics[
                        "processed_samples"
                    ]
                ),
                "correct": int(
                    global_metrics[
                        "correct_predictions"
                    ]
                ),
                "predicted_as_class": "",
                "true_positive": "",
                "false_positive": "",
                "false_negative": "",
                "true_negative": "",
                "class_accuracy": "",
                "precision": "",
                "recall": "",
                "f1_score": "",
                "overall_accuracy": (
                    global_metrics[
                        "overall_accuracy"
                    ]
                ),
                "checkpoint_epoch": (
                    checkpoint_epoch
                ),
                "checkpoint_validation_loss": (
                    checkpoint_validation_loss
                ),
                "checkpoint_validation_accuracy": (
                    checkpoint_validation_accuracy
                ),
            }
        )

        # Une ligne par classe.
        for row in rows:

            writer.writerow(
                {
                    "scope": "CLASS",
                    **row,
                    "overall_accuracy": (
                        global_metrics[
                            "overall_accuracy"
                        ]
                    ),
                    "checkpoint_epoch": (
                        checkpoint_epoch
                    ),
                    "checkpoint_validation_loss": (
                        checkpoint_validation_loss
                    ),
                    "checkpoint_validation_accuracy": (
                        checkpoint_validation_accuracy
                    ),
                }
            )


# ================================================================
# SAUVEGARDE DES MATRICES EN CSV
# ================================================================


def save_confusion_matrix_csv(
    confusion_matrix: np.ndarray,
    output_path: Path,
    decimal_places: Optional[int] = None,
) -> None:
    """
    Sauvegarde une matrice de confusion dans un fichier CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "true_class",
                *FASHION_MNIST_CLASSES,
            ]
        )

        for class_index, class_name in enumerate(
            FASHION_MNIST_CLASSES
        ):

            values = confusion_matrix[
                class_index
            ]

            if decimal_places is None:

                formatted_values = [
                    int(value)
                    for value in values
                ]

            else:

                formatted_values = [
                    round(
                        float(value),
                        decimal_places,
                    )
                    for value in values
                ]

            writer.writerow(
                [
                    class_name,
                    *formatted_values,
                ]
            )


# ================================================================
# FIGURE DE LA MATRICE DE CONFUSION
# ================================================================


def save_confusion_matrix_figure(
    normalized_confusion_matrix: np.ndarray,
    output_path: Path,
) -> None:
    """
    Sauvegarde une représentation graphique de la matrice
    de confusion normalisée.
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
        "Fashion-MNIST - Matrice de confusion normalisée\n"
        "Classifieur indépendant - validation"
    )

    axis.set_xlabel(
        "Classe prédite"
    )

    axis.set_ylabel(
        "Classe réelle"
    )

    class_indices = np.arange(
        len(FASHION_MNIST_CLASSES)
    )

    axis.set_xticks(
        class_indices
    )

    axis.set_yticks(
        class_indices
    )

    axis.set_xticklabels(
        FASHION_MNIST_CLASSES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        FASHION_MNIST_CLASSES
    )

    # Affiche la valeur dans chaque cellule.
    for true_index in range(
        len(FASHION_MNIST_CLASSES)
    ):

        for predicted_index in range(
            len(FASHION_MNIST_CLASSES)
        ):

            value = (
                normalized_confusion_matrix[
                    true_index,
                    predicted_index,
                ]
            )

            axis.text(
                predicted_index,
                true_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ================================================================
# AFFICHAGE
# ================================================================


def print_results(
    global_metrics: dict[str, float],
    per_class_metrics: list[
        dict[str, float | int | str]
    ],
    checkpoint: dict,
) -> None:
    """
    Affiche les résultats de façon lisible.
    """

    print("=" * 94)

    print(
        "ÉVALUATION DU CLASSIFIEUR "
        "FASHION-MNIST SUR LA VALIDATION"
    )

    print("=" * 94)

    print(
        f"Époque du checkpoint        : "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Loss validation checkpoint  : "
        f"{checkpoint.get('best_validation_loss'):.6f}"
    )

    print(
        f"Accuracy checkpoint         : "
        f"{checkpoint.get('validation_accuracy'):.6f}"
    )

    print(
        f"Images évaluées             : "
        f"{int(global_metrics['processed_samples'])}"
    )

    print(
        f"Prédictions correctes       : "
        f"{int(global_metrics['correct_predictions'])}"
    )

    print(
        f"Accuracy globale            : "
        f"{global_metrics['overall_accuracy']:.4%}"
    )

    print("=" * 94)

    header = (
        f"{'ID':>2}  "
        f"{'Classe':<14} "
        f"{'Support':>8} "
        f"{'Correct':>8} "
        f"{'Acc classe':>11} "
        f"{'Précision':>10} "
        f"{'Rappel':>10} "
        f"{'F1':>10}"
    )

    print(
        header
    )

    print("-" * 94)

    for row in per_class_metrics:

        print(
            f"{int(row['class_index']):>2}  "
            f"{str(row['class_name']):<14} "
            f"{int(row['support']):>8} "
            f"{int(row['correct']):>8} "
            f"{float(row['class_accuracy']):>10.2%} "
            f"{float(row['precision']):>9.2%} "
            f"{float(row['recall']):>9.2%} "
            f"{float(row['f1_score']):>9.2%}"
        )

    print("=" * 94)


# ================================================================
# ARGUMENTS
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments disponibles dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Évaluer le classifieur Fashion-MNIST "
            "sur les 6 000 images de validation."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Checkpoint du classifieur. "
            f"Par défaut : {DEFAULT_CHECKPOINT}"
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Seed utilisé pour reconstruire le split "
            "54k / 6k. Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Nombre d'images par batch. "
            "Valeur par défaut : 128."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Nombre de workers du DataLoader. "
            "Sous Windows, conserver 0."
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
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed global utilisé pendant l'évaluation. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help=(
            "CSV contenant les métriques globales "
            "et par classe."
        ),
    )

    parser.add_argument(
        "--confusion-csv",
        type=Path,
        default=DEFAULT_CONFUSION_CSV,
        help=(
            "CSV contenant la matrice de confusion brute."
        ),
    )

    parser.add_argument(
        "--confusion-normalized-csv",
        type=Path,
        default=DEFAULT_CONFUSION_NORMALIZED_CSV,
        help=(
            "CSV contenant la matrice de confusion normalisée."
        ),
    )

    parser.add_argument(
        "--confusion-figure",
        type=Path,
        default=DEFAULT_CONFUSION_FIGURE,
        help=(
            "Image PNG de la matrice de confusion normalisée."
        ),
    )

    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help=(
            "Limite temporairement le nombre de batchs "
            "pour un smoke test."
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

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
        )

    if (
        args.max_eval_batches is not None
        and args.max_eval_batches <= 0
    ):
        raise ValueError(
            "max-eval-batches doit être strictement positif."
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

    pin_memory = (
        device.type == "cuda"
    )

    # On réutilise exactement la fonction déjà utilisée pendant
    # l'entraînement des VAE/CVAE et du classifieur.
    #
    # Le premier DataLoader retourné correspond au train.
    # Il n'est pas utilisé ici.
    _, validation_loader = create_dataloaders(
        batch_size=args.batch_size,
        seed=args.split_seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model, checkpoint = load_classifier(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    is_smoke_test = (
        args.max_eval_batches
        is not None
    )

    print("=" * 94)

    print(
        "PRÉPARATION DE L'ÉVALUATION "
        "DU CLASSIFIEUR"
    )

    print("=" * 94)

    print(
        f"Checkpoint                  : "
        f"{args.checkpoint}"
    )

    print(
        f"Appareil utilisé            : "
        f"{device}"
    )

    print(
        f"Seed du split               : "
        f"{args.split_seed}"
    )

    print(
        f"Images disponibles          : "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Taille des batchs           : "
        f"{args.batch_size}"
    )

    print(
        "Split évalué               : "
        "validation"
    )

    print(
        "Jeu officiel de test       : "
        "NON UTILISÉ"
    )

    if is_smoke_test:

        print(
            f"Mode                        : "
            f"TEST RAPIDE "
            f"({args.max_eval_batches} batchs)"
        )

    print("=" * 94)

    global_metrics, confusion_matrix = evaluate_classifier(
        model=model,
        dataloader=validation_loader,
        device=device,
        max_batches=(
            args.max_eval_batches
        ),
    )

    per_class_metrics = (
        calculate_per_class_metrics(
            confusion_matrix
        )
    )

    normalized_confusion_matrix = (
        normalize_confusion_matrix(
            confusion_matrix
        )
    )

    print_results(
        global_metrics=global_metrics,
        per_class_metrics=(
            per_class_metrics
        ),
        checkpoint=checkpoint,
    )

    save_metrics_csv(
        rows=per_class_metrics,
        global_metrics=global_metrics,
        checkpoint=checkpoint,
        output_path=args.metrics_csv,
    )

    save_confusion_matrix_csv(
        confusion_matrix=confusion_matrix,
        output_path=args.confusion_csv,
    )

    save_confusion_matrix_csv(
        confusion_matrix=(
            normalized_confusion_matrix
        ),
        output_path=(
            args.confusion_normalized_csv
        ),
        decimal_places=6,
    )

    save_confusion_matrix_figure(
        normalized_confusion_matrix=(
            normalized_confusion_matrix
        ),
        output_path=(
            args.confusion_figure
        ),
    )

    print(
        f"Métriques CSV               : "
        f"{args.metrics_csv}"
    )

    print(
        f"Matrice brute CSV           : "
        f"{args.confusion_csv}"
    )

    print(
        f"Matrice normalisée CSV      : "
        f"{args.confusion_normalized_csv}"
    )

    print(
        f"Figure matrice              : "
        f"{args.confusion_figure}"
    )

    print("=" * 94)


if __name__ == "__main__":
    main()