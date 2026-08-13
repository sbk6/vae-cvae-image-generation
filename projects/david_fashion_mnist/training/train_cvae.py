"""
Entraînement du CVAE sur Fashion-MNIST avec suivi MLflow.

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
6. la sauvegarde de l'historique CSV ;
7. l'enregistrement des paramètres, métriques et artefacts
   avec MLflow.

Le jeu officiel de test de 10 000 images n'est pas utilisé
pendant l'entraînement.

Exemple :

    python -m training.train_cvae --beta 1 --epochs 20

Smoke test :

    python -m training.train_cvae \
        --beta 1 \
        --epochs 2 \
        --max-train-batches 2 \
        --max-val-batches 1 \
        --run-name cvae_mlflow_smoke
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

# Ce fichier se trouve dans :
#
# projects/david_fashion_mnist/training/train_cvae.py
#
# parents[1] correspond donc au sous-projet Fashion-MNIST.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

HISTORY_DIR = (
    PROJECT_ROOT
    / "results"
    / "training_histories"
)


# ================================================================
# ENTRAÎNEMENT
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

    Contrairement au VAE classique, le CVAE reçoit à la fois :

        - les images ;
        - les labels.

    Parameters
    ----------
    model:
        Modèle CVAE.

    dataloader:
        DataLoader d'entraînement.

    optimizer:
        Optimiseur PyTorch.

    device:
        CPU ou CUDA.

    beta:
        Poids appliqué au terme KL.

    max_batches:
        Nombre maximum optionnel de batchs.

        Cette option sert uniquement aux smoke tests.

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

    progress_total = len(dataloader)

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

    for batch_index, (images, labels) in progress_bar:

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

        # La fonction de perte reste la même que pour le VAE.
        #
        # Le conditionnement est réalisé dans l'architecture du CVAE,
        # pas directement dans la loss.
        total_loss, reconstruction_loss, kl_loss = vae_loss(
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

        processed_samples += batch_size

        progress_bar.set_postfix(
            total=f"{total_loss.item():.2f}",
            reconstruction=f"{reconstruction_loss.item():.2f}",
            kl=f"{kl_loss.item():.2f}",
        )

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        reconstruction_loss_sum=(
            reconstruction_loss_sum
        ),
        kl_loss_sum=kl_loss_sum,
        processed_samples=processed_samples,
    )


# ================================================================
# VALIDATION
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

    progress_total = len(dataloader)

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

    for batch_index, (images, labels) in progress_bar:

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

        reconstruction, mu, logvar, _ = model(
            images,
            labels,
        )

        total_loss, reconstruction_loss, kl_loss = vae_loss(
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

        processed_samples += batch_size

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        reconstruction_loss_sum=(
            reconstruction_loss_sum
        ),
        kl_loss_sum=kl_loss_sum,
        processed_samples=processed_samples,
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
# MLFLOW
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
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": str(device),
        "smoke_test": is_smoke_test,
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

    # Les tags sont utiles pour filtrer rapidement les runs
    # dans l'interface MLflow.
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


def log_epoch_to_mlflow(
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """
    Enregistre dans MLflow les métriques d'une époque.
    """

    mlflow.log_metrics(
        {
            "train_total": (
                train_metrics["total"]
            ),
            "train_reconstruction": (
                train_metrics["reconstruction"]
            ),
            "train_kl": (
                train_metrics["kl"]
            ),
            "validation_total": (
                validation_metrics["total"]
            ),
            "validation_reconstruction": (
                validation_metrics["reconstruction"]
            ),
            "validation_kl": (
                validation_metrics["kl"]
            ),
        },
        step=epoch,
    )


# ================================================================
# ARGUMENTS DU TERMINAL
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments du script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner un CVAE sur Fashion-MNIST avec MLflow."
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=(
            "Nombre d'époques. "
            "Valeur par défaut actuelle : 20."
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Graine aléatoire. "
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

    # Ces deux arguments sont réservés aux tests rapides.
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

    if args.epochs <= 0:
        raise ValueError(
            "epochs doit être "
            "strictement positif."
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

    parser = build_argument_parser()

    args = parser.parse_args()

    validate_arguments(
        args
    )

    # Reproductibilité.
    set_random_seed(
        args.seed
    )

    # CPU ou GPU.
    device = select_device(
        args.device
    )

    pin_memory = (
        device.type == "cuda"
    )

    # Utilise exactement la même séparation train / validation
    # que le script VAE.
    train_loader, validation_loader = create_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Création du CVAE.
    model = CVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Un run limité en batchs est considéré comme un smoke test.
    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    # Construction du nom du run.
    if args.run_name is not None:
        run_name = args.run_name

    else:
        run_name = (
            f"cvae_beta_{beta_to_tag(args.beta)}"
        )

        if is_smoke_test:
            run_name += "_smoke"

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{run_name}.pt"
    )

    history_path = (
        HISTORY_DIR
        / f"{run_name}_history.csv"
    )

    # Configuration sauvegardée dans le checkpoint.
    configuration = {
        "dataset": "Fashion-MNIST",
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    # Réutilise exactement le même backend MLflow SQLite
    # que le VAE.
    tracking_uri = configure_mlflow(
        tracking_uri=(
            args.mlflow_tracking_uri
        ),
        experiment_name=(
            args.mlflow_experiment_name
        ),
    )

    print("=" * 72)
    print(
        "ENTRAÎNEMENT DU CVAE "
        "SUR FASHION-MNIST + MLFLOW"
    )
    print("=" * 72)

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
        f"Nombre d'époques            : "
        f"{args.epochs}"
    )

    print(
        f"Taux d'apprentissage        : "
        f"{args.learning_rate}"
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

    if args.mlflow_tracking_uri is None:
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

    print("=" * 72)

    history: list[
        dict[str, float]
    ] = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    training_start_time = (
        time.perf_counter()
    )

    # Une exécution du script = un run MLflow.
    with mlflow.start_run(
        run_name=run_name,
    ) as active_run:

        print(
            f"MLflow run ID               : "
            f"{active_run.info.run_id}"
        )

        print("=" * 72)

        log_mlflow_parameters(
            args=args,
            device=device,
            is_smoke_test=is_smoke_test,
        )

        for epoch in range(
            1,
            args.epochs + 1,
        ):

            # ----------------------------
            # ENTRAÎNEMENT
            # ----------------------------

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=args.beta,
                max_batches=(
                    args.max_train_batches
                ),
            )

            # ----------------------------
            # VALIDATION
            # ----------------------------

            validation_metrics = validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                device=device,
                beta=args.beta,
                max_batches=(
                    args.max_val_batches
                ),
            )

            # ----------------------------
            # HISTORIQUE
            # ----------------------------

            history_row = {
                "epoch": epoch,

                "train_total": (
                    train_metrics["total"]
                ),

                "train_reconstruction": (
                    train_metrics[
                        "reconstruction"
                    ]
                ),

                "train_kl": (
                    train_metrics["kl"]
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
                f"{epoch:02d}/{args.epochs:02d} | "

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

            # ----------------------------
            # MLFLOW : MÉTRIQUES
            # ----------------------------

            log_epoch_to_mlflow(
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=(
                    validation_metrics
                ),
            )

            # ----------------------------
            # MEILLEUR CHECKPOINT
            # ----------------------------

            if (
                validation_metrics["total"]
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_metrics[
                        "total"
                    ]
                )

                best_epoch = epoch

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    beta=args.beta,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                    configuration=configuration,
                )

                print(
                    "  -> Nouveau meilleur modèle "
                    f"sauvegardé à l'époque {epoch}."
                )

            # ----------------------------
            # CSV LOCAL
            # ----------------------------

            save_history_csv(
                history=history,
                path=history_path,
            )

        # ========================================================
        # FIN DE L'ENTRAÎNEMENT
        # ========================================================

        training_duration_seconds = (
            time.perf_counter()
            - training_start_time
        )

        # Résumé du run.
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

                "training_duration_seconds": (
                    training_duration_seconds
                ),
            }
        )

        # Checkpoint comme artefact MLflow.
        if checkpoint_path.exists():

            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoints",
            )

        # Historique CSV comme artefact MLflow.
        if history_path.exists():

            mlflow.log_artifact(
                str(history_path),
                artifact_path=(
                    "training_histories"
                ),
            )

        # Tags utiles dans l'interface.
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

        print("=" * 72)
        print(
            "ENTRAÎNEMENT TERMINÉ"
        )
        print("=" * 72)

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

        if args.mlflow_tracking_uri is None:

            print(
                f"Base SQLite MLflow          : "
                f"{MLFLOW_DB_PATH}"
            )

            print(
                f"Artefacts MLflow            : "
                f"{MLFLOW_ARTIFACT_DIR}"
            )

        print("=" * 72)


if __name__ == "__main__":
    main()