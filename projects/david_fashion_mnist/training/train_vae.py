"""
Entraînement du VAE sur Fashion-MNIST avec suivi MLflow.

Ce script réalise les opérations suivantes :

1. charge Fashion-MNIST ;
2. sépare les 60 000 images officielles d'entraînement :
       - 54 000 images pour l'entraînement ;
       - 6 000 images pour la validation ;
3. entraîne le VAE ;
4. mesure les performances sur la validation après chaque époque ;
5. sauvegarde le meilleur checkpoint ;
6. sauvegarde l'historique des pertes dans un fichier CSV ;
7. enregistre les paramètres, métriques et artefacts avec MLflow.

Le jeu officiel de test de 10 000 images n'est pas utilisé pendant
l'entraînement.

Exemple :

    python -m training.train_vae --beta 1 --epochs 20

Smoke test MLflow :

    python -m training.train_vae \
        --beta 1 \
        --epochs 2 \
        --max-train-batches 2 \
        --max-val-batches 1 \
        --run-name vae_mlflow_smoke
"""

from __future__ import annotations

import argparse
import csv
import platform
import random
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vae import VAE
from training.losses import vae_loss


# ================================================================
# CHEMINS DU SOUS-PROJET
# ================================================================

# Ce fichier se trouve dans :
#
# projects/david_fashion_mnist/training/train_vae.py
#
# parents[1] correspond donc à :
#
# projects/david_fashion_mnist
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_DIR = PROJECT_ROOT / "results" / "training_histories"

# MLflow 3.15 n'utilise plus par défaut le FileStore comme backend
# de tracking.
#
# Nous utilisons donc :
#
# - SQLite pour les métadonnées, paramètres et métriques ;
# - un dossier local pour les artefacts.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

DEFAULT_MLFLOW_EXPERIMENT = "fashion_mnist_vae_cvae"


# ================================================================
# REPRODUCTIBILITÉ
# ================================================================


def set_random_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires pour favoriser la reproductibilité.

    La graine contrôle :

    - le module random de Python ;
    - NumPy ;
    - PyTorch ;
    - CUDA lorsqu'il est disponible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested_device: str) -> torch.device:
    """
    Sélectionne le CPU ou CUDA.

    Parameters
    ----------
    requested_device:
        "auto"
            Utilise CUDA lorsqu'il est disponible, sinon le CPU.

        "cpu"
            Force le CPU.

        "cuda"
            Force CUDA.
    """

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA a été demandé, mais aucune carte graphique "
            "compatible ou aucune version CUDA de PyTorch "
            "n'est disponible."
        )

    return torch.device(requested_device)


# ================================================================
# DONNÉES
# ================================================================


def create_dataloaders(
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """
    Charge Fashion-MNIST et crée les DataLoader train et validation.

    Fashion-MNIST contient officiellement :

        - 60 000 images d'entraînement ;
        - 10 000 images de test.

    Dans ce script, les 60 000 images officielles d'entraînement sont
    séparées en :

        - 54 000 images d'entraînement ;
        - 6 000 images de validation.

    Le jeu officiel de test n'est pas chargé ici.
    """

    transform = transforms.ToTensor()

    full_train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    train_size = 54_000
    validation_size = 6_000

    if train_size + validation_size != len(full_train_dataset):
        raise RuntimeError(
            "La taille attendue de Fashion-MNIST est différente "
            f"de la taille observée : {len(full_train_dataset)}."
        )

    # Garantit que le split train/validation reste identique
    # pour une même seed.
    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        dataset=full_train_dataset,
        lengths=[train_size, validation_size],
        generator=split_generator,
    )

    # Contrôle également l'ordre des batchs d'entraînement.
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=loader_generator,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, validation_loader


# ================================================================
# MÉTRIQUES
# ================================================================


def calculate_average_metrics(
    total_loss_sum: float,
    reconstruction_loss_sum: float,
    kl_loss_sum: float,
    processed_samples: int,
) -> dict[str, float]:
    """
    Calcule les pertes moyennes par image.
    """

    if processed_samples <= 0:
        raise RuntimeError(
            "Aucune image n'a été traitée pendant cette époque."
        )

    return {
        "total": total_loss_sum / processed_samples,
        "reconstruction": (
            reconstruction_loss_sum / processed_samples
        ),
        "kl": kl_loss_sum / processed_samples,
    }


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
    Entraîne le VAE pendant une époque.

    Returns
    -------
    dict[str, float]
        Contient :

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
        desc="Entraînement VAE",
        leave=False,
    )

    for batch_index, (images, _) in progress_bar:
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device=device,
            non_blocking=True,
        )

        batch_size = images.shape[0]

        optimizer.zero_grad(
            set_to_none=True,
        )

        reconstruction, mu, logvar, _ = model(images)

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
            total_loss.item() * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )

        kl_loss_sum += (
            kl_loss.item() * batch_size
        )

        processed_samples += batch_size

        progress_bar.set_postfix(
            total=f"{total_loss.item():.2f}",
            reconstruction=f"{reconstruction_loss.item():.2f}",
            kl=f"{kl_loss.item():.2f}",
        )

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        reconstruction_loss_sum=reconstruction_loss_sum,
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
    Évalue le VAE sur le jeu de validation.

    Pendant la validation :

    - aucun gradient n'est calculé ;
    - aucun poids n'est modifié.
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
        desc="Validation VAE",
        leave=False,
    )

    for batch_index, (images, _) in progress_bar:
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(
            device=device,
            non_blocking=True,
        )

        batch_size = images.shape[0]

        reconstruction, mu, logvar, _ = model(images)

        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        total_loss_sum += (
            total_loss.item() * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )

        kl_loss_sum += (
            kl_loss.item() * batch_size
        )

        processed_samples += batch_size

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        reconstruction_loss_sum=reconstruction_loss_sum,
        kl_loss_sum=kl_loss_sum,
        processed_samples=processed_samples,
    )


# ================================================================
# CHECKPOINT
# ================================================================


def beta_to_tag(beta: float) -> str:
    """
    Transforme beta en texte utilisable dans un nom de fichier.

    Exemples
    --------
    0.1 -> "01"
    1.0 -> "1"
    4.0 -> "4"
    """

    return f"{beta:g}".replace(".", "")


def save_checkpoint(
    path: Path,
    model: VAE,
    optimizer: Optimizer,
    epoch: int,
    beta: float,
    best_validation_loss: float,
    configuration: dict,
) -> None:
    """
    Sauvegarde le meilleur checkpoint du VAE.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "VAE",
        "epoch": epoch,
        "beta": beta,
        "best_validation_loss": best_validation_loss,
        "configuration": configuration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
    Sauvegarde les métriques de chaque époque dans un fichier CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_total",
        "train_reconstruction",
        "train_kl",
        "validation_total",
        "validation_reconstruction",
        "validation_kl",
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
        writer.writerows(history)


# ================================================================
# MLFLOW
# ================================================================


def build_default_mlflow_tracking_uri() -> str:
    """
    Construit l'URI SQLite locale utilisée par MLflow.

    Sous Windows, on obtient par exemple :

        sqlite:///D:/projet/mlflow.db

    Sous Linux / Colab, on obtiendra par exemple :

        sqlite:////content/projet/mlflow.db

    La même fonction reste donc portable entre Windows et Colab.
    """

    database_path = (
        MLFLOW_DB_PATH.resolve().as_posix()
    )

    return f"sqlite:///{database_path}"


def configure_mlflow(
    tracking_uri: Optional[str],
    experiment_name: str,
) -> str:
    """
    Configure le backend de suivi MLflow.

    Cas 1
    -----
    Aucun tracking URI n'est fourni.

    Le projet utilise alors :

        SQLite :
            mlflow.db

        Artefacts :
            mlartifacts/

    Cas 2
    -----
    Un tracking URI est explicitement fourni dans le terminal.

    MLflow utilise directement cet URI.

    Cette possibilité sera notamment utile si l'on décide plus tard
    d'utiliser un serveur MLflow ou un environnement Colab.
    """

    if tracking_uri is None:
        resolved_tracking_uri = (
            build_default_mlflow_tracking_uri()
        )

        MLFLOW_ARTIFACT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        mlflow.set_tracking_uri(
            resolved_tracking_uri
        )

        # Vérifie si l'expérience existe déjà dans la base SQLite.
        experiment = mlflow.get_experiment_by_name(
            experiment_name
        )

        # Lors de la toute première exécution, on crée l'expérience
        # et on précise où ses artefacts doivent être sauvegardés.
        if experiment is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=(
                    MLFLOW_ARTIFACT_DIR.resolve().as_uri()
                ),
            )

        mlflow.set_experiment(
            experiment_name
        )

    else:
        resolved_tracking_uri = tracking_uri

        mlflow.set_tracking_uri(
            resolved_tracking_uri
        )

        # Avec un serveur ou un backend externe, sa configuration
        # détermine l'emplacement des artefacts.
        mlflow.set_experiment(
            experiment_name
        )

    return resolved_tracking_uri


def log_mlflow_parameters(
    args: argparse.Namespace,
    device: torch.device,
    is_smoke_test: bool,
) -> None:
    """
    Enregistre les hyperparamètres et informations principales.
    """

    parameters = {
        "dataset": "Fashion-MNIST",
        "model_type": "VAE",
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
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

    mlflow.set_tags(
        {
            "dataset": "Fashion-MNIST",
            "model_type": "VAE",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
        }
    )


def log_epoch_to_mlflow(
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """
    Enregistre les six métriques principales d'une époque.
    """

    mlflow.log_metrics(
        {
            "train_total": train_metrics["total"],
            "train_reconstruction": (
                train_metrics["reconstruction"]
            ),
            "train_kl": train_metrics["kl"],
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
# ARGUMENTS
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments utilisables dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner un VAE sur Fashion-MNIST avec MLflow."
        )
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Poids du terme KL. Valeur par défaut : 1.",
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
        help="Nombre d'images par batch. Valeur par défaut : 128.",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Dimension de l'espace latent. Valeur par défaut : 16.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Dimension cachée. Valeur par défaut : 256.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage. Valeur par défaut : 0.001.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire. Valeur par défaut : 42.",
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
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Appareil de calcul. Valeur par défaut : auto.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nom personnalisé du run MLflow.",
    )

    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help=(
            "Nom de l'expérience MLflow. "
            f"Par défaut : {DEFAULT_MLFLOW_EXPERIMENT}."
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

    # Utilisés uniquement pour les smoke tests.
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limite temporaire des batchs d'entraînement.",
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limite temporaire des batchs de validation.",
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Vérifie les paramètres fournis dans le terminal.
    """

    if args.beta < 0:
        raise ValueError(
            "beta doit être supérieur ou égal à zéro."
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs doit être strictement positif."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être strictement positif."
        )

    if args.latent_dim <= 0:
        raise ValueError(
            "latent-dim doit être strictement positif."
        )

    if args.hidden_dim <= 0:
        raise ValueError(
            "hidden-dim doit être strictement positif."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
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
    Point d'entrée principal de l'entraînement du VAE.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)

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

    model = VAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    if args.run_name is not None:
        run_name = args.run_name

    else:
        run_name = (
            f"vae_beta_{beta_to_tag(args.beta)}"
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

    configuration = {
        "dataset": "Fashion-MNIST",
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    # Configure SQLite / serveur MLflow avant de créer le run.
    tracking_uri = configure_mlflow(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
    )

    print("=" * 72)
    print("ENTRAÎNEMENT DU VAE SUR FASHION-MNIST + MLFLOW")
    print("=" * 72)
    print(f"Appareil utilisé            : {device}")
    print(f"Images d'entraînement       : {len(train_loader.dataset)}")
    print(f"Images de validation        : {len(validation_loader.dataset)}")
    print(f"Taille des batchs           : {args.batch_size}")
    print(f"Dimension latente           : {args.latent_dim}")
    print(f"Beta                        : {args.beta}")
    print(f"Nombre d'époques            : {args.epochs}")
    print(f"Taux d'apprentissage        : {args.learning_rate}")
    print(f"Nom du run MLflow           : {run_name}")
    print(f"Expérience MLflow           : {args.mlflow_experiment_name}")
    print(f"Tracking URI MLflow         : {tracking_uri}")

    if args.mlflow_tracking_uri is None:
        print(f"Base SQLite MLflow          : {MLFLOW_DB_PATH}")
        print(f"Artefacts MLflow            : {MLFLOW_ARTIFACT_DIR}")

    if is_smoke_test:
        print("Mode                        : TEST RAPIDE")

    print("=" * 72)

    history: list[dict[str, float]] = []

    best_validation_loss = float("inf")
    best_epoch = 0

    training_start_time = time.perf_counter()

    # Une exécution du script correspond à un run MLflow.
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
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=args.beta,
                max_batches=args.max_train_batches,
            )

            validation_metrics = validate_one_epoch(
                model=model,
                dataloader=validation_loader,
                device=device,
                beta=args.beta,
                max_batches=args.max_val_batches,
            )

            history_row = {
                "epoch": epoch,
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
            }

            history.append(
                history_row
            )

            print(
                f"Époque {epoch:02d}/{args.epochs:02d} | "
                f"Train total={train_metrics['total']:.4f} | "
                f"Train recon="
                f"{train_metrics['reconstruction']:.4f} | "
                f"Train KL={train_metrics['kl']:.4f} | "
                f"Val total="
                f"{validation_metrics['total']:.4f} | "
                f"Val recon="
                f"{validation_metrics['reconstruction']:.4f} | "
                f"Val KL="
                f"{validation_metrics['kl']:.4f}"
            )

            # Enregistre les courbes train/validation dans MLflow.
            log_epoch_to_mlflow(
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )

            # Sauvegarde du meilleur checkpoint.
            if (
                validation_metrics["total"]
                < best_validation_loss
            ):
                best_validation_loss = (
                    validation_metrics["total"]
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

            # Sauvegarde locale de l'historique après chaque époque.
            save_history_csv(
                history=history,
                path=history_path,
            )

        training_duration_seconds = (
            time.perf_counter()
            - training_start_time
        )

        # Métriques finales du run.
        mlflow.log_metrics(
            {
                "best_epoch": float(best_epoch),
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

        # Sauvegarde des fichiers importants comme artefacts MLflow.
        if checkpoint_path.exists():
            mlflow.log_artifact(
                str(checkpoint_path),
                artifact_path="checkpoints",
            )

        if history_path.exists():
            mlflow.log_artifact(
                str(history_path),
                artifact_path="training_histories",
            )

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
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 72)
        print(
            f"MLflow run ID               : "
            f"{active_run.info.run_id}"
        )
        print(f"Meilleure époque            : {best_epoch}")
        print(
            "Meilleure loss validation   : "
            f"{best_validation_loss:.4f}"
        )
        print(
            "Durée d'entraînement        : "
            f"{training_duration_seconds:.2f} secondes"
        )
        print(f"Checkpoint                  : {checkpoint_path}")
        print(f"Historique CSV              : {history_path}")
        print(f"MLflow tracking URI         : {tracking_uri}")

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