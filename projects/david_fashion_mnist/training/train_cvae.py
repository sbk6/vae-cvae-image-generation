"""
Entraînement du CVAE sur Fashion-MNIST avec MLflow et early stopping.

Le CVAE est un VAE conditionnel. Il reçoit :

    - une image ;
    - son label de classe.

Fashion-MNIST possède 10 classes :

    0 : T-shirt/top
    1 : Trouser
    2 : Pullover
    3 : Dress
    4 : Coat
    5 : Sandal
    6 : Shirt
    7 : Sneaker
    8 : Bag
    9 : Ankle boot

Ce script réalise :

1. le chargement de Fashion-MNIST ;
2. la séparation des 60 000 images officielles d'entraînement :
       - 54 000 images pour l'entraînement ;
       - 6 000 images pour la validation ;
3. l'entraînement du CVAE ;
4. la validation après chaque époque ;
5. la sauvegarde du meilleur checkpoint ;
6. l'early stopping basé sur la loss totale de validation ;
7. la sauvegarde de l'historique CSV ;
8. l'enregistrement des paramètres, métriques et artefacts
   avec MLflow.

Le jeu officiel de test de 10 000 images n'est pas utilisé
pendant l'entraînement.

Reproductibilité
----------------
Deux graines sont distinguées :

    split_seed
        Contrôle uniquement la séparation fixe train / validation.

    seed
        Contrôle les aléas d'entraînement :
        - initialisation du modèle ;
        - ordre des batchs ;
        - générateurs aléatoires Python / NumPy / PyTorch.

Cette séparation permet de réaliser les expériences multi-seed sur
exactement le même jeu de validation.

Configuration finale prévue :

    max_epochs = 100
    patience   = 10
    min_delta  = 0.0
    split_seed = 42

Exemple :

    python -m training.train_cvae --beta 1

Exemple multi-seed :

    python -m training.train_cvae \
        --beta 1 \
        --split-seed 42 \
        --seed 123 \
        --max-epochs 100 \
        --patience 10 \
        --min-delta 0

L'ancien argument --epochs reste accepté pour compatibilité :

    python -m training.train_cvae --beta 1 --epochs 100
"""

from __future__ import annotations

import argparse
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

from models.cvae import CVAE
from training.losses import vae_loss

# Plusieurs fonctions générales sont partagées avec le VAE.
#
# Cela garantit notamment que VAE et CVAE utilisent :
#
# - exactement le même split train / validation ;
# - la même gestion des seeds ;
# - la même sélection CPU / GPU ;
# - la même configuration MLflow ;
# - la même méthode de calcul des moyennes.
from training.train_vae import (
    DEFAULT_MLFLOW_EXPERIMENT,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_PATH,
    beta_to_tag,
    calculate_average_metrics,
    configure_mlflow,
    create_dataloaders,
    save_history_csv,
    select_device,
    set_random_seed,
)


# ================================================================
# CHEMINS DU SOUS-PROJET
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

HISTORY_DIR = (
    PROJECT_ROOT
    / "results"
    / "training_histories"
)


# ================================================================
# ENTRAÎNEMENT D'UNE ÉPOQUE
# ================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    beta: float,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Entraîne le CVAE pendant une époque.

    Contrairement au VAE classique, le CVAE reçoit :

        - les images ;
        - les labels.

    max_batches est utilisé uniquement pour les smoke tests.

    Returns
    -------
    dict[str, float]
        Pertes moyennes par image :

        - total ;
        - reconstruction ;
        - kl.
    """

    model.train()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0

    processed_samples = 0

    progress_total = len(
        dataloader
    )

    if max_batches is not None:
        progress_total = min(
            progress_total,
            max_batches,
        )

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Entraînement CVAE",
        leave=False,
    )

    for (
        batch_index,
        (images, labels),
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

        # Le CVAE reçoit l'image ET le label.
        reconstruction, mu, logvar, _ = model(
            images,
            labels,
        )

        # La fonction de perte est la même que pour le VAE.
        #
        # Le conditionnement est réalisé dans l'architecture
        # du CVAE, pas directement dans la loss.
        (
            total_loss,
            reconstruction_loss,
            kl_loss,
        ) = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        total_loss.backward()

        optimizer.step()

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item()
            * batch_size
        )

        kl_loss_sum += (
            kl_loss.item()
            * batch_size
        )

        processed_samples += (
            batch_size
        )

        progress_bar.set_postfix(
            total=(
                f"{total_loss.item():.2f}"
            ),
            reconstruction=(
                f"{reconstruction_loss.item():.2f}"
            ),
            kl=(
                f"{kl_loss.item():.2f}"
            ),
        )

    return calculate_average_metrics(
        total_loss_sum=(
            total_loss_sum
        ),
        reconstruction_loss_sum=(
            reconstruction_loss_sum
        ),
        kl_loss_sum=(
            kl_loss_sum
        ),
        processed_samples=(
            processed_samples
        ),
    )


# ================================================================
# VALIDATION D'UNE ÉPOQUE
# ================================================================


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Évalue le CVAE sur le jeu de validation.

    Pendant cette étape :

    - aucun gradient n'est calculé ;
    - aucun poids n'est modifié ;
    - les labels sont toujours fournis au CVAE.
    """

    model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0

    processed_samples = 0

    progress_total = len(
        dataloader
    )

    if max_batches is not None:
        progress_total = min(
            progress_total,
            max_batches,
        )

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Validation CVAE",
        leave=False,
    )

    for (
        batch_index,
        (images, labels),
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

        batch_size = (
            images.shape[0]
        )

        reconstruction, mu, logvar, _ = model(
            images,
            labels,
        )

        (
            total_loss,
            reconstruction_loss,
            kl_loss,
        ) = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item()
            * batch_size
        )

        kl_loss_sum += (
            kl_loss.item()
            * batch_size
        )

        processed_samples += (
            batch_size
        )

    return calculate_average_metrics(
        total_loss_sum=(
            total_loss_sum
        ),
        reconstruction_loss_sum=(
            reconstruction_loss_sum
        ),
        kl_loss_sum=(
            kl_loss_sum
        ),
        processed_samples=(
            processed_samples
        ),
    )


# ================================================================
# CHECKPOINT
# ================================================================


def save_checkpoint(
    path: Path,
    model: CVAE,
    optimizer: Optimizer,
    epoch: int,
    beta: float,
    best_validation_loss: float,
    configuration: dict,
) -> None:
    """
    Sauvegarde le meilleur checkpoint du CVAE.

    Le fichier contient :

    - le type du modèle ;
    - l'époque ;
    - beta ;
    - la meilleure loss de validation ;
    - la configuration ;
    - les poids du modèle ;
    - l'état de l'optimiseur.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "CVAE",
        "epoch": epoch,
        "beta": beta,
        "best_validation_loss": (
            best_validation_loss
        ),
        "configuration": (
            configuration
        ),
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
# MLFLOW : PARAMÈTRES
# ================================================================


def log_mlflow_parameters(
    args: argparse.Namespace,
    device: torch.device,
    is_smoke_test: bool,
) -> None:
    """
    Enregistre les paramètres du CVAE dans MLflow.
    """

    parameters = {
        "dataset": "Fashion-MNIST",
        "model_type": "CVAE",
        "beta": (
            args.beta
        ),

        "max_epochs": (
            args.max_epochs
        ),

        "batch_size": (
            args.batch_size
        ),
        "latent_dim": (
            args.latent_dim
        ),
        "hidden_dim": (
            args.hidden_dim
        ),
        "num_classes": (
            args.num_classes
        ),
        "learning_rate": (
            args.learning_rate
        ),

        # --------------------------------------------------------
        # REPRODUCTIBILITÉ
        # --------------------------------------------------------
        #
        # seed est conservé pour la compatibilité avec les anciens
        # runs.
        #
        # training_seed rend désormais son rôle explicite.
        #
        # split_seed identifie la séparation train / validation.
        # --------------------------------------------------------
        "seed": (
            args.seed
        ),
        "training_seed": (
            args.seed
        ),
        "split_seed": (
            args.split_seed
        ),

        "num_workers": (
            args.num_workers
        ),
        "device": str(
            device
        ),

        # Paramètres de l'early stopping.
        "early_stopping": True,
        "patience": (
            args.patience
        ),
        "min_delta": (
            args.min_delta
        ),
        "early_stopping_monitor": (
            "validation_total"
        ),

        "smoke_test": (
            is_smoke_test
        ),

        "max_train_batches": (
            args.max_train_batches
            if args.max_train_batches is not None
            else "None"
        ),

        "max_val_batches": (
            args.max_val_batches
            if args.max_val_batches is not None
            else "None"
        ),
    }

    mlflow.log_params(
        parameters
    )

    mlflow.set_tags(
        {
            "dataset": "Fashion-MNIST",
            "model_type": "CVAE",
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
    train_metrics: dict[
        str,
        float,
    ],
    validation_metrics: dict[
        str,
        float,
    ],
    best_validation_loss: float,
    epochs_without_improvement: int,
    is_improvement: bool,
) -> None:
    """
    Enregistre les métriques d'une époque dans MLflow.

    En plus des losses, on suit l'état de l'early stopping.
    """

    mlflow.log_metrics(
        {
            "train_total": (
                train_metrics[
                    "total"
                ]
            ),

            "train_reconstruction": (
                train_metrics[
                    "reconstruction"
                ]
            ),

            "train_kl": (
                train_metrics[
                    "kl"
                ]
            ),

            "validation_total": (
                validation_metrics[
                    "total"
                ]
            ),

            "validation_reconstruction": (
                validation_metrics[
                    "reconstruction"
                ]
            ),

            "validation_kl": (
                validation_metrics[
                    "kl"
                ]
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
# ARGUMENTS DU TERMINAL
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments utilisables dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner un CVAE sur Fashion-MNIST "
            "avec MLflow et early stopping."
        )
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help=(
            "Poids du terme KL. "
            "Valeur par défaut : 1."
        ),
    )

    # --epochs reste accepté pour assurer la compatibilité
    # avec les anciennes commandes du projet.
    parser.add_argument(
        "--max-epochs",
        "--epochs",
        dest="max_epochs",
        type=int,
        default=100,
        help=(
            "Nombre maximal d'époques. "
            "Valeur par défaut : 100. "
            "--epochs reste accepté comme alias."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Nombre d'époques consécutives sans amélioration "
            "avant l'arrêt. Valeur par défaut : 10."
        ),
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help=(
            "Amélioration minimale exigée de la loss de validation. "
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
        "--latent-dim",
        type=int,
        default=16,
        help=(
            "Dimension de l'espace latent. "
            "Valeur par défaut : 16."
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help=(
            "Dimension de la couche cachée. "
            "Valeur par défaut : 256."
        ),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help=(
            "Nombre de classes. "
            "Fashion-MNIST en possède 10."
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

    # ------------------------------------------------------------
    # SEED D'ENTRAÎNEMENT
    # ------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed d'entraînement : initialisation du modèle, "
            "ordre des batchs et autres aléas d'entraînement. "
            "Valeur par défaut : 42."
        ),
    )

    # ------------------------------------------------------------
    # SEED DU SPLIT
    # ------------------------------------------------------------

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Seed utilisée uniquement pour la séparation fixe "
            "54 000 train / 6 000 validation. "
            "Pour les expériences multi-seed, conserver 42. "
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
        default=None,
        help=(
            "Nom personnalisé du run MLflow."
        ),
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help=(
            "Nom de l'expérience MLflow. "
            f"Par défaut : "
            f"{DEFAULT_MLFLOW_EXPERIMENT}."
        ),
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help=(
            "URI MLflow optionnel. "
            "Sans valeur, SQLite local est utilisé."
        ),
    )

    # Réservés aux smoke tests.
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Limite temporaire du nombre "
            "de batchs d'entraînement."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "Limite temporaire du nombre "
            "de batchs de validation."
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
    Vérifie les valeurs reçues dans le terminal.
    """

    if args.beta < 0:
        raise ValueError(
            "beta doit être supérieur "
            "ou égal à zéro."
        )

    if args.max_epochs <= 0:
        raise ValueError(
            "max-epochs doit être "
            "strictement positif."
        )

    if args.patience <= 0:
        raise ValueError(
            "patience doit être "
            "strictement positive."
        )

    if args.min_delta < 0:
        raise ValueError(
            "min-delta doit être "
            "supérieur ou égal à zéro."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être "
            "strictement positif."
        )

    if args.latent_dim <= 0:
        raise ValueError(
            "latent-dim doit être "
            "strictement positif."
        )

    if args.hidden_dim <= 0:
        raise ValueError(
            "hidden-dim doit être "
            "strictement positif."
        )

    if args.num_classes <= 1:
        raise ValueError(
            "num-classes doit être "
            "supérieur à un."
        )

    if args.num_classes != 10:
        raise ValueError(
            "Fashion-MNIST contient "
            "exactement 10 classes."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate doit être "
            "strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
        )

    if not args.mlflow_experiment_name.strip():
        raise ValueError(
            "mlflow-experiment-name "
            "ne peut pas être vide."
        )

    if (
        args.max_train_batches is not None
        and args.max_train_batches <= 0
    ):
        raise ValueError(
            "max-train-batches doit être "
            "strictement positif."
        )

    if (
        args.max_val_batches is not None
        and args.max_val_batches <= 0
    ):
        raise ValueError(
            "max-val-batches doit être "
            "strictement positif."
        )


# ================================================================
# PROGRAMME PRINCIPAL
# ================================================================


def main() -> None:
    """
    Point d'entrée principal de l'entraînement du CVAE.
    """

    parser = (
        build_argument_parser()
    )

    args = (
        parser.parse_args()
    )

    validate_arguments(
        args
    )

    # ------------------------------------------------------------
    # SEED D'ENTRAÎNEMENT
    # ------------------------------------------------------------
    #
    # Cette seed peut varier entre les runs multi-seed.
    # ------------------------------------------------------------

    set_random_seed(
        args.seed
    )

    # CPU ou GPU.
    device = select_device(
        args.device
    )

    pin_memory = (
        device.type
        == "cuda"
    )

    # ------------------------------------------------------------
    # DATASETS ET DATALOADERS
    # ------------------------------------------------------------
    #
    # split_seed :
    #   reste fixe afin que VAE et CVAE ainsi que tous les runs
    #   multi-seed soient évalués sur les mêmes 6 000 images.
    #
    # loader_seed :
    #   suit args.seed et fait partie des aléas d'entraînement.
    #
    # seed :
    #   reste transmis pour préserver la rétrocompatibilité.
    # ------------------------------------------------------------

    (
        train_loader,
        validation_loader,
    ) = create_dataloaders(
        batch_size=(
            args.batch_size
        ),
        seed=(
            args.seed
        ),
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            pin_memory
        ),
        split_seed=(
            args.split_seed
        ),
        loader_seed=(
            args.seed
        ),
    )

    # Création du CVAE.
    model = CVAE(
        latent_dim=(
            args.latent_dim
        ),
        hidden_dim=(
            args.hidden_dim
        ),
        num_classes=(
            args.num_classes
        ),
    ).to(
        device
    )

    optimizer = Adam(
        model.parameters(),
        lr=(
            args.learning_rate
        ),
    )

    # Une limitation en batchs indique un smoke test.
    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    # Construction du nom du run.
    if args.run_name is not None:

        run_name = (
            args.run_name
        )

    else:

        run_name = (
            f"cvae_beta_"
            f"{beta_to_tag(args.beta)}"
        )

        if is_smoke_test:
            run_name += (
                "_smoke"
            )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{run_name}.pt"
    )

    history_path = (
        HISTORY_DIR
        / f"{run_name}_history.csv"
    )

    # ------------------------------------------------------------
    # CONFIGURATION SAUVEGARDÉE DANS LE CHECKPOINT
    # ------------------------------------------------------------
    #
    # "epochs" et "seed" restent présents pour assurer la
    # compatibilité avec les anciens checkpoints et scripts.
    #
    # Les nouvelles clés explicites sont :
    #
    # training_seed
    # split_seed
    # ------------------------------------------------------------

    configuration = {
        "dataset": "Fashion-MNIST",
        "beta": (
            args.beta
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
        "latent_dim": (
            args.latent_dim
        ),
        "hidden_dim": (
            args.hidden_dim
        ),
        "num_classes": (
            args.num_classes
        ),
        "learning_rate": (
            args.learning_rate
        ),

        # Compatibilité avec les anciens outils.
        "seed": (
            args.seed
        ),

        # Nouvelles clés méthodologiques.
        "training_seed": (
            args.seed
        ),
        "split_seed": (
            args.split_seed
        ),

        "num_workers": (
            args.num_workers
        ),
        "device": str(
            device
        ),
    }

    # Réutilise le même backend MLflow SQLite que le VAE.
    tracking_uri = configure_mlflow(
        tracking_uri=(
            args.mlflow_tracking_uri
        ),
        experiment_name=(
            args.mlflow_experiment_name
        ),
    )

    print(
        "=" * 74
    )

    print(
        "ENTRAÎNEMENT DU CVAE SUR "
        "FASHION-MNIST + MLFLOW + EARLY STOPPING"
    )

    print(
        "=" * 74
    )

    print(
        f"Appareil utilisé            : "
        f"{device}"
    )

    print(
        f"Images d'entraînement       : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Images de validation        : "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Taille des batchs           : "
        f"{args.batch_size}"
    )

    print(
        f"Dimension latente           : "
        f"{args.latent_dim}"
    )

    print(
        f"Nombre de classes           : "
        f"{args.num_classes}"
    )

    print(
        f"Beta                        : "
        f"{args.beta}"
    )

    print(
        f"Nombre maximal d'époques    : "
        f"{args.max_epochs}"
    )

    print(
        f"Patience                    : "
        f"{args.patience}"
    )

    print(
        f"Min delta                   : "
        f"{args.min_delta}"
    )

    print(
        f"Taux d'apprentissage        : "
        f"{args.learning_rate}"
    )

    print(
        f"Seed d'entraînement         : "
        f"{args.seed}"
    )

    print(
        f"Seed du split               : "
        f"{args.split_seed}"
    )

    print(
        f"Nom du run MLflow           : "
        f"{run_name}"
    )

    print(
        f"Expérience MLflow           : "
        f"{args.mlflow_experiment_name}"
    )

    print(
        f"Tracking URI MLflow         : "
        f"{tracking_uri}"
    )

    if (
        args.mlflow_tracking_uri
        is None
    ):

        print(
            f"Base SQLite MLflow          : "
            f"{MLFLOW_DB_PATH}"
        )

        print(
            f"Artefacts MLflow            : "
            f"{MLFLOW_ARTIFACT_DIR}"
        )

    if is_smoke_test:

        print(
            "Mode                        : "
            "TEST RAPIDE"
        )

    print(
        "=" * 74
    )

    history: list[
        dict[str, float]
    ] = []

    # Meilleure loss de validation reconnue selon le critère
    # de l'early stopping.
    best_validation_loss = (
        float("inf")
    )

    best_epoch = 0

    # Nombre d'époques consécutives sans amélioration suffisante.
    epochs_without_improvement = 0

    early_stopping_triggered = (
        False
    )

    stopped_epoch = 0

    training_start_time = (
        time.perf_counter()
    )

    # Une exécution du script correspond à un run MLflow.
    with mlflow.start_run(
        run_name=(
            run_name
        ),
    ) as active_run:

        print(
            f"MLflow run ID               : "
            f"{active_run.info.run_id}"
        )

        print(
            "=" * 74
        )

        log_mlflow_parameters(
            args=args,
            device=device,
            is_smoke_test=(
                is_smoke_test
            ),
        )

        for epoch in range(
            1,
            args.max_epochs + 1,
        ):

            # ====================================================
            # ENTRAÎNEMENT
            # ====================================================

            train_metrics = (
                train_one_epoch(
                    model=model,
                    dataloader=(
                        train_loader
                    ),
                    optimizer=(
                        optimizer
                    ),
                    device=device,
                    beta=(
                        args.beta
                    ),
                    max_batches=(
                        args.max_train_batches
                    ),
                )
            )

            # ====================================================
            # VALIDATION
            # ====================================================

            validation_metrics = (
                validate_one_epoch(
                    model=model,
                    dataloader=(
                        validation_loader
                    ),
                    device=device,
                    beta=(
                        args.beta
                    ),
                    max_batches=(
                        args.max_val_batches
                    ),
                )
            )

            current_validation_loss = (
                validation_metrics[
                    "total"
                ]
            )

            # ====================================================
            # HISTORIQUE CSV
            # ====================================================

            history_row = {
                "epoch": epoch,

                "train_total": (
                    train_metrics[
                        "total"
                    ]
                ),

                "train_reconstruction": (
                    train_metrics[
                        "reconstruction"
                    ]
                ),

                "train_kl": (
                    train_metrics[
                        "kl"
                    ]
                ),

                "validation_total": (
                    validation_metrics[
                        "total"
                    ]
                ),

                "validation_reconstruction": (
                    validation_metrics[
                        "reconstruction"
                    ]
                ),

                "validation_kl": (
                    validation_metrics[
                        "kl"
                    ]
                ),
            }

            history.append(
                history_row
            )

            print(
                f"Époque "
                f"{epoch:03d}/"
                f"{args.max_epochs:03d} | "

                f"Train total="
                f"{train_metrics['total']:.4f} | "

                f"Train recon="
                f"{train_metrics['reconstruction']:.4f} | "

                f"Train KL="
                f"{train_metrics['kl']:.4f} | "

                f"Val total="
                f"{validation_metrics['total']:.4f} | "

                f"Val recon="
                f"{validation_metrics['reconstruction']:.4f} | "

                f"Val KL="
                f"{validation_metrics['kl']:.4f}"
            )

            # ====================================================
            # EARLY STOPPING
            # ====================================================

            # Une amélioration est reconnue lorsque :
            #
            # nouvelle_loss < meilleure_loss - min_delta
            #
            # Avec min_delta = 0.0, toute baisse compte comme
            # une amélioration.
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

                best_epoch = (
                    epoch
                )

                # Une amélioration remet le compteur à zéro.
                epochs_without_improvement = (
                    0
                )

                save_checkpoint(
                    path=(
                        checkpoint_path
                    ),
                    model=model,
                    optimizer=(
                        optimizer
                    ),
                    epoch=(
                        epoch
                    ),
                    beta=(
                        args.beta
                    ),
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    configuration=(
                        configuration
                    ),
                )

                print(
                    "  -> Nouveau meilleur modèle "
                    f"sauvegardé à l'époque {epoch}."
                )

            else:

                epochs_without_improvement += (
                    1
                )

                print(
                    "  -> Pas d'amélioration suffisante. "
                    f"Compteur early stopping : "
                    f"{epochs_without_improvement}/"
                    f"{args.patience}"
                )

            # ====================================================
            # MLFLOW : MÉTRIQUES DE L'ÉPOQUE
            # ====================================================

            log_epoch_to_mlflow(
                epoch=(
                    epoch
                ),
                train_metrics=(
                    train_metrics
                ),
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
            # CSV LOCAL
            # ====================================================

            # On sauvegarde le CSV avant la décision d'arrêt afin
            # de conserver aussi la dernière époque exécutée.
            save_history_csv(
                history=(
                    history
                ),
                path=(
                    history_path
                ),
            )

            # ====================================================
            # DÉCISION D'ARRÊT
            # ====================================================

            if (
                epochs_without_improvement
                >= args.patience
            ):

                early_stopping_triggered = (
                    True
                )

                stopped_epoch = (
                    epoch
                )

                print(
                    "=" * 74
                )

                print(
                    "EARLY STOPPING DÉCLENCHÉ"
                )

                print(
                    f"Aucune amélioration suffisante "
                    f"pendant {args.patience} "
                    f"époques consécutives."
                )

                print(
                    f"Arrêt à l'époque            : "
                    f"{stopped_epoch}"
                )

                print(
                    f"Meilleure époque            : "
                    f"{best_epoch}"
                )

                print(
                    f"Meilleure loss validation   : "
                    f"{best_validation_loss:.4f}"
                )

                print(
                    "=" * 74
                )

                break

        # Si l'early stopping ne s'est pas déclenché,
        # l'entraînement est allé jusqu'au nombre maximal d'époques.
        if not early_stopping_triggered:

            stopped_epoch = len(
                history
            )

        training_duration_seconds = (
            time.perf_counter()
            - training_start_time
        )

        # ========================================================
        # MLFLOW : MÉTRIQUES FINALES
        # ========================================================

        mlflow.log_metrics(
            {
                "best_epoch": float(
                    best_epoch
                ),

                "best_validation_loss": (
                    best_validation_loss
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
        # MLFLOW : ARTEFACTS
        # ========================================================

        if checkpoint_path.exists():

            mlflow.log_artifact(
                str(
                    checkpoint_path
                ),
                artifact_path=(
                    "checkpoints"
                ),
            )

        if history_path.exists():

            mlflow.log_artifact(
                str(
                    history_path
                ),
                artifact_path=(
                    "training_histories"
                ),
            )

        # ========================================================
        # MLFLOW : TAGS
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

        # ========================================================
        # AFFICHAGE FINAL
        # ========================================================

        print(
            "=" * 74
        )

        print(
            "ENTRAÎNEMENT TERMINÉ"
        )

        print(
            "=" * 74
        )

        print(
            f"MLflow run ID               : "
            f"{active_run.info.run_id}"
        )

        print(
            f"Meilleure époque            : "
            f"{best_epoch}"
        )

        print(
            f"Meilleure loss validation   : "
            f"{best_validation_loss:.4f}"
        )

        print(
            f"Époques réellement exécutées: "
            f"{len(history)}"
        )

        print(
            f"Époque d'arrêt              : "
            f"{stopped_epoch}"
        )

        print(
            f"Early stopping déclenché    : "
            f"{early_stopping_triggered}"
        )

        print(
            f"Seed d'entraînement         : "
            f"{args.seed}"
        )

        print(
            f"Seed du split               : "
            f"{args.split_seed}"
        )

        print(
            f"Durée d'entraînement        : "
            f"{training_duration_seconds:.2f} secondes"
        )

        print(
            f"Checkpoint                  : "
            f"{checkpoint_path}"
        )

        print(
            f"Historique CSV              : "
            f"{history_path}"
        )

        print(
            f"MLflow tracking URI         : "
            f"{tracking_uri}"
        )

        if (
            args.mlflow_tracking_uri
            is None
        ):

            print(
                f"Base SQLite MLflow          : "
                f"{MLFLOW_DB_PATH}"
            )

            print(
                f"Artefacts MLflow            : "
                f"{MLFLOW_ARTIFACT_DIR}"
            )

        print(
            "=" * 74
        )


if __name__ == "__main__":
    main()