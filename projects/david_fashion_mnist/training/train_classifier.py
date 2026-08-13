"""
Entraînement d'un classifieur indépendant sur Fashion-MNIST.

Ce classifieur n'est pas utilisé pour générer des images.

Son objectif est d'apprendre à reconnaître les dix classes de
Fashion-MNIST afin de servir ensuite d'évaluateur externe pour
mesurer la cohérence conditionnelle des images générées par le CVAE.

Le protocole utilisé est le même que pour les VAE/CVAE :

- 54 000 images pour l'entraînement ;
- 6 000 images pour la validation ;
- aucune utilisation du jeu officiel de test pendant cette phase.

Le meilleur checkpoint est sélectionné selon la loss de validation.

Exemple
-------

    python -m training.train_classifier

Exemple avec CUDA :

    python -m training.train_classifier --device cuda

Smoke test :

    python -m training.train_classifier \
        --max-epochs 3 \
        --max-train-batches 2 \
        --max-val-batches 2
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fashion_classifier import (
    FASHION_MNIST_CLASSES,
    FashionMNISTClassifier,
)
from training.train_vae import (
    CHECKPOINT_DIR,
    DEFAULT_MLFLOW_EXPERIMENT,
    HISTORY_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_PATH,
    configure_mlflow,
    create_dataloaders,
    select_device,
    set_random_seed,
)


# ================================================================
# CONFIGURATION
# ================================================================

DEFAULT_CLASSIFIER_EXPERIMENT = (
    "fashion_mnist_classifier"
)

DEFAULT_RUN_NAME = (
    "fashion_classifier_seed42"
)


# ================================================================
# MÉTRIQUES
# ================================================================


def calculate_average_metrics(
    total_loss_sum: float,
    correct_predictions: int,
    processed_samples: int,
) -> dict[str, float]:
    """
    Calcule la loss moyenne et l'accuracy.

    Parameters
    ----------
    total_loss_sum:
        Somme pondérée des losses des batchs.

    correct_predictions:
        Nombre total de prédictions correctes.

    processed_samples:
        Nombre total d'images traitées.

    Returns
    -------
    metrics:
        Dictionnaire contenant :

        - loss ;
        - accuracy.
    """

    if processed_samples <= 0:
        raise RuntimeError(
            "Aucune image n'a été traitée."
        )

    average_loss = (
        total_loss_sum
        / processed_samples
    )

    accuracy = (
        correct_predictions
        / processed_samples
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
    }


# ================================================================
# ENTRAÎNEMENT D'UNE ÉPOQUE
# ================================================================


def train_one_epoch(
    model: FashionMNISTClassifier,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Entraîne le classifieur pendant une époque.

    max_batches est utilisé uniquement pour les smoke tests.
    """

    model.train()

    total_loss_sum = 0.0
    correct_predictions = 0
    processed_samples = 0

    progress_total = len(dataloader)

    if max_batches is not None:
        progress_total = min(
            progress_total,
            max_batches,
        )

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Entraînement classifieur",
        leave=False,
    )

    for batch_index, (
        images,
        labels,
    ) in progress_bar:

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

        batch_size = images.shape[0]

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

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

        total_loss_sum += (
            loss.item()
            * batch_size
        )

        processed_samples += batch_size

        current_accuracy = (
            correct_predictions
            / processed_samples
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{current_accuracy:.4f}",
        )

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        correct_predictions=correct_predictions,
        processed_samples=processed_samples,
    )


# ================================================================
# VALIDATION D'UNE ÉPOQUE
# ================================================================


@torch.no_grad()
def validate_one_epoch(
    model: FashionMNISTClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Évalue le classifieur sur la validation.

    Aucun gradient n'est calculé.
    Aucun poids n'est modifié.
    """

    model.eval()

    total_loss_sum = 0.0
    correct_predictions = 0
    processed_samples = 0

    progress_total = len(dataloader)

    if max_batches is not None:
        progress_total = min(
            progress_total,
            max_batches,
        )

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Validation classifieur",
        leave=False,
    )

    for batch_index, (
        images,
        labels,
    ) in progress_bar:

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

        batch_size = images.shape[0]

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
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

        total_loss_sum += (
            loss.item()
            * batch_size
        )

        processed_samples += batch_size

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        correct_predictions=correct_predictions,
        processed_samples=processed_samples,
    )


# ================================================================
# CHECKPOINT
# ================================================================


def save_checkpoint(
    path: Path,
    model: FashionMNISTClassifier,
    optimizer: Optimizer,
    epoch: int,
    best_validation_loss: float,
    validation_accuracy: float,
    configuration: dict,
) -> None:
    """
    Sauvegarde le meilleur checkpoint du classifieur.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "FashionMNISTClassifier",
        "epoch": epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "validation_accuracy": (
            validation_accuracy
        ),
        "configuration": configuration,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


# ================================================================
# HISTORIQUE CSV
# ================================================================


def save_history_csv(
    history: list[dict[str, float]],
    path: Path,
) -> None:
    """
    Sauvegarde les métriques de chaque époque dans un CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]

    with path.open(
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
            history
        )


# ================================================================
# MLFLOW : PARAMÈTRES
# ================================================================


def log_mlflow_parameters(
    args: argparse.Namespace,
    device: torch.device,
    is_smoke_test: bool,
) -> None:
    """
    Enregistre la configuration du classifieur dans MLflow.
    """

    mlflow.log_params(
        {
            "dataset": "Fashion-MNIST",
            "model_type": (
                "FashionMNISTClassifier"
            ),
            "num_classes": 10,
            "max_epochs": (
                args.max_epochs
            ),
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "seed": (
                args.seed
            ),
            "num_workers": (
                args.num_workers
            ),
            "device": str(
                device
            ),
            "loss_function": (
                "CrossEntropyLoss"
            ),
            "optimizer": "Adam",
            "early_stopping": True,
            "patience": (
                args.patience
            ),
            "min_delta": (
                args.min_delta
            ),
            "early_stopping_monitor": (
                "validation_loss"
            ),
            "train_size": 54_000,
            "validation_size": 6_000,
            "test_used": False,
            "smoke_test": (
                is_smoke_test
            ),
            "max_train_batches": (
                args.max_train_batches
                if args.max_train_batches
                is not None
                else "None"
            ),
            "max_val_batches": (
                args.max_val_batches
                if args.max_val_batches
                is not None
                else "None"
            ),
        }
    )

    mlflow.set_tags(
        {
            "dataset": "Fashion-MNIST",
            "model_type": (
                "FashionMNISTClassifier"
            ),
            "role": (
                "independent_conditional_evaluator"
            ),
            "python_version": (
                platform.python_version()
            ),
            "torch_version": (
                torch.__version__
            ),
            "mlflow_version": (
                mlflow.__version__
            ),
        }
    )


# ================================================================
# MLFLOW : MÉTRIQUES PAR ÉPOQUE
# ================================================================


def log_epoch_to_mlflow(
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    best_validation_loss: float,
    epochs_without_improvement: int,
    is_improvement: bool,
) -> None:
    """
    Enregistre les métriques d'une époque dans MLflow.
    """

    mlflow.log_metrics(
        {
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "best_validation_loss_so_far": (
                best_validation_loss
            ),
            "epochs_without_improvement": float(
                epochs_without_improvement
            ),
            "is_improvement": (
                1.0
                if is_improvement
                else 0.0
            ),
        },
        step=epoch,
    )


# ================================================================
# ARGUMENTS
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments du script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner un classifieur indépendant "
            "sur Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--max-epochs",
        "--epochs",
        dest="max_epochs",
        type=int,
        default=50,
        help=(
            "Nombre maximal d'époques. "
            "Valeur par défaut : 50."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Nombre d'époques consécutives "
            "sans amélioration avant l'arrêt. "
            "Valeur par défaut : 10."
        ),
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help=(
            "Amélioration minimale exigée "
            "de la loss de validation. "
            "Valeur par défaut : 0.0."
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
        "--learning-rate",
        type=float,
        default=0.001,
        help=(
            "Taux d'apprentissage. "
            "Valeur par défaut : 0.001."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Graine utilisée pour le split, "
            "l'initialisation et l'ordre des batchs. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Nombre de processus de chargement. "
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
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=(
            "Nom du run MLflow et du checkpoint."
        ),
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_CLASSIFIER_EXPERIMENT,
        help=(
            "Nom de l'expérience MLflow."
        ),
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "URI MLflow optionnelle. "
            "Sans valeur, la base SQLite locale "
            "du projet est utilisée."
        ),
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Limite le nombre de batchs d'entraînement "
            "pour un smoke test."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "Limite le nombre de batchs de validation "
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

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs doit être strictement positif."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience doit être strictement positive."
        )

    if args.min_delta < 0:
        raise ValueError(
            "min-delta doit être supérieur ou égal à zéro."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être strictement positif."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
        )

    if not args.run_name.strip():
        raise ValueError(
            "run-name ne peut pas être vide."
        )

    if not args.mlflow_experiment_name.strip():
        raise ValueError(
            "mlflow-experiment-name ne peut pas être vide."
        )

    if (
        args.max_train_batches is not None
        and args.max_train_batches <= 0
    ):
        raise ValueError(
            "max-train-batches doit être strictement positif."
        )

    if (
        args.max_val_batches is not None
        and args.max_val_batches <= 0
    ):
        raise ValueError(
            "max-val-batches doit être strictement positif."
        )


# ================================================================
# PROGRAMME PRINCIPAL
# ================================================================


def main() -> None:
    """
    Point d'entrée principal de l'entraînement du classifieur.
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

    train_loader, validation_loader = create_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = FashionMNISTClassifier(
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    run_name = (
        args.run_name
    )

    if is_smoke_test:
        run_name = (
            f"{run_name}_smoke"
        )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{run_name}.pt"
    )

    history_path = (
        HISTORY_DIR
        / f"{run_name}_history.csv"
    )

    configuration = {
        "dataset": "Fashion-MNIST",
        "model_type": (
            "FashionMNISTClassifier"
        ),
        "num_classes": 10,
        "class_names": (
            list(FASHION_MNIST_CLASSES)
        ),
        "epochs": (
            args.max_epochs
        ),
        "max_epochs": (
            args.max_epochs
        ),
        "patience": (
            args.patience
        ),
        "min_delta": (
            args.min_delta
        ),
        "batch_size": (
            args.batch_size
        ),
        "learning_rate": (
            args.learning_rate
        ),
        "seed": (
            args.seed
        ),
        "num_workers": (
            args.num_workers
        ),
        "device": str(
            device
        ),
        "train_size": 54_000,
        "validation_size": 6_000,
        "test_used": False,
    }

    tracking_uri = configure_mlflow(
        tracking_uri=(
            args.mlflow_tracking_uri
        ),
        experiment_name=(
            args.mlflow_experiment_name
        ),
    )

    print("=" * 78)

    print(
        "ENTRAÎNEMENT DU CLASSIFIEUR "
        "FASHION-MNIST + MLFLOW + EARLY STOPPING"
    )

    print("=" * 78)

    print(
        f"Appareil utilisé              : "
        f"{device}"
    )

    print(
        f"Images d'entraînement         : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Images de validation          : "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Taille des batchs             : "
        f"{args.batch_size}"
    )

    print(
        f"Nombre de classes             : "
        f"{len(FASHION_MNIST_CLASSES)}"
    )

    print(
        f"Nombre maximal d'époques      : "
        f"{args.max_epochs}"
    )

    print(
        f"Patience                      : "
        f"{args.patience}"
    )

    print(
        f"Min delta                     : "
        f"{args.min_delta}"
    )

    print(
        f"Taux d'apprentissage          : "
        f"{args.learning_rate}"
    )

    print(
        f"Seed                          : "
        f"{args.seed}"
    )

    print(
        f"Nom du run MLflow             : "
        f"{run_name}"
    )

    print(
        f"Expérience MLflow             : "
        f"{args.mlflow_experiment_name}"
    )

    print(
        f"Tracking URI MLflow           : "
        f"{tracking_uri}"
    )

    if args.mlflow_tracking_uri is None:

        print(
            f"Base SQLite MLflow            : "
            f"{MLFLOW_DB_PATH}"
        )

        print(
            f"Artefacts MLflow              : "
            f"{MLFLOW_ARTIFACT_DIR}"
        )

    print(
        "Jeu officiel de test utilisé  : "
        "NON"
    )

    if is_smoke_test:

        print(
            "Mode                          : "
            "TEST RAPIDE"
        )

    print("=" * 78)

    history: list[
        dict[str, float]
    ] = []

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    early_stopping_triggered = False

    stopped_epoch = 0

    training_start_time = (
        time.perf_counter()
    )

    with mlflow.start_run(
        run_name=run_name,
    ) as active_run:

        print(
            f"MLflow run ID                 : "
            f"{active_run.info.run_id}"
        )

        print("=" * 78)

        log_mlflow_parameters(
            args=args,
            device=device,
            is_smoke_test=is_smoke_test,
        )

        for epoch in range(
            1,
            args.max_epochs + 1,
        ):

            # ====================================================
            # ENTRAÎNEMENT
            # ====================================================

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_batches=(
                    args.max_train_batches
                ),
            )

            # ====================================================
            # VALIDATION
            # ====================================================

            validation_metrics = validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                max_batches=(
                    args.max_val_batches
                ),
            )

            current_validation_loss = (
                validation_metrics["loss"]
            )

            current_validation_accuracy = (
                validation_metrics["accuracy"]
            )

            # ====================================================
            # HISTORIQUE
            # ====================================================

            history_row = {
                "epoch": epoch,
                "train_loss": (
                    train_metrics["loss"]
                ),
                "train_accuracy": (
                    train_metrics["accuracy"]
                ),
                "validation_loss": (
                    current_validation_loss
                ),
                "validation_accuracy": (
                    current_validation_accuracy
                ),
            }

            history.append(
                history_row
            )

            print(
                f"Époque "
                f"{epoch:03d}/{args.max_epochs:03d} | "
                f"Train loss="
                f"{train_metrics['loss']:.4f} | "
                f"Train acc="
                f"{train_metrics['accuracy']:.4f} | "
                f"Val loss="
                f"{current_validation_loss:.4f} | "
                f"Val acc="
                f"{current_validation_accuracy:.4f}"
            )

            # ====================================================
            # EARLY STOPPING
            # ====================================================

            is_improvement = (
                current_validation_loss
                < (
                    best_validation_loss
                    - args.min_delta
                )
            )

            if is_improvement:

                best_validation_loss = (
                    current_validation_loss
                )

                best_validation_accuracy = (
                    current_validation_accuracy
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    validation_accuracy=(
                        best_validation_accuracy
                    ),
                    configuration=configuration,
                )

                print(
                    "  -> Nouveau meilleur classifieur "
                    f"sauvegardé à l'époque {epoch}."
                )

            else:

                epochs_without_improvement += 1

                print(
                    "  -> Pas d'amélioration suffisante. "
                    f"Compteur early stopping : "
                    f"{epochs_without_improvement}/"
                    f"{args.patience}"
                )

            # ====================================================
            # MLFLOW
            # ====================================================

            log_epoch_to_mlflow(
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=(
                    validation_metrics
                ),
                best_validation_loss=(
                    best_validation_loss
                ),
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                is_improvement=(
                    is_improvement
                ),
            )

            # ====================================================
            # CSV
            # ====================================================

            save_history_csv(
                history=history,
                path=history_path,
            )

            # ====================================================
            # ARRÊT
            # ====================================================

            if (
                epochs_without_improvement
                >= args.patience
            ):

                early_stopping_triggered = True

                stopped_epoch = epoch

                print("=" * 78)

                print(
                    "EARLY STOPPING DÉCLENCHÉ"
                )

                print(
                    f"Arrêt à l'époque              : "
                    f"{stopped_epoch}"
                )

                print(
                    f"Meilleure époque              : "
                    f"{best_epoch}"
                )

                print(
                    f"Meilleure loss validation     : "
                    f"{best_validation_loss:.4f}"
                )

                print(
                    f"Accuracy au meilleur checkpoint: "
                    f"{best_validation_accuracy:.4f}"
                )

                print("=" * 78)

                break

        if not early_stopping_triggered:

            stopped_epoch = len(
                history
            )

        training_duration_seconds = (
            time.perf_counter()
            - training_start_time
        )

        # ========================================================
        # RÉSUMÉ FINAL MLFLOW
        # ========================================================

        mlflow.log_metrics(
            {
                "best_epoch": float(
                    best_epoch
                ),
                "best_validation_loss": (
                    best_validation_loss
                ),
                "best_checkpoint_validation_accuracy": (
                    best_validation_accuracy
                ),
                "epochs_completed": float(
                    len(history)
                ),
                "stopped_epoch": float(
                    stopped_epoch
                ),
                "early_stopping_triggered": (
                    1.0
                    if early_stopping_triggered
                    else 0.0
                ),
                "final_epochs_without_improvement": float(
                    epochs_without_improvement
                ),
                "training_duration_seconds": (
                    training_duration_seconds
                ),
            }
        )

        # ========================================================
        # ARTEFACTS MLFLOW
        # ========================================================

        if checkpoint_path.exists():

            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoints",
            )

        if history_path.exists():

            mlflow.log_artifact(
                str(history_path),
                artifact_path=(
                    "training_histories"
                ),
            )

        # ========================================================
        # TAGS MLFLOW
        # ========================================================

        mlflow.set_tag(
            "best_checkpoint",
            checkpoint_path.name,
        )

        mlflow.set_tag(
            "run_status",
            (
                "smoke_test"
                if is_smoke_test
                else "training"
            ),
        )

        mlflow.set_tag(
            "early_stopping_triggered",
            str(
                early_stopping_triggered
            ),
        )

        mlflow.set_tag(
            "stopping_reason",
            (
                "early_stopping"
                if early_stopping_triggered
                else "max_epochs_reached"
            ),
        )

        mlflow.set_tag(
            "test_used",
            "False",
        )

        # ========================================================
        # AFFICHAGE FINAL
        # ========================================================

        print("=" * 78)

        print(
            "ENTRAÎNEMENT DU CLASSIFIEUR TERMINÉ"
        )

        print("=" * 78)

        print(
            f"MLflow run ID                 : "
            f"{active_run.info.run_id}"
        )

        print(
            f"Meilleure époque              : "
            f"{best_epoch}"
        )

        print(
            f"Meilleure loss validation     : "
            f"{best_validation_loss:.4f}"
        )

        print(
            f"Accuracy au meilleur checkpoint: "
            f"{best_validation_accuracy:.4f}"
        )

        print(
            f"Époques réellement exécutées  : "
            f"{len(history)}"
        )

        print(
            f"Époque d'arrêt                : "
            f"{stopped_epoch}"
        )

        print(
            f"Early stopping déclenché      : "
            f"{early_stopping_triggered}"
        )

        print(
            f"Durée d'entraînement          : "
            f"{training_duration_seconds:.2f} secondes"
        )

        print(
            f"Checkpoint                    : "
            f"{checkpoint_path}"
        )

        print(
            f"Historique CSV                : "
            f"{history_path}"
        )

        print(
            f"MLflow tracking URI           : "
            f"{tracking_uri}"
        )

        if args.mlflow_tracking_uri is None:

            print(
                f"Base SQLite MLflow            : "
                f"{MLFLOW_DB_PATH}"
            )

            print(
                f"Artefacts MLflow              : "
                f"{MLFLOW_ARTIFACT_DIR}"
            )

        print(
            "Jeu officiel de test utilisé  : "
            "NON"
        )

        print("=" * 78)


if __name__ == "__main__":
    main()