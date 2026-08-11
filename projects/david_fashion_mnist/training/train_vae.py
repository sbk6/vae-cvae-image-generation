"""
Entraînement du VAE sur Fashion-MNIST.

Ce script réalise les opérations suivantes :

1. téléchargement ou chargement de Fashion-MNIST ;
2. séparation des 60 000 images officielles d'entraînement :
       - 54 000 images pour l'entraînement ;
       - 6 000 images pour la validation ;
3. entraînement du VAE ;
4. évaluation après chaque époque sur la validation ;
5. sauvegarde du meilleur modèle ;
6. sauvegarde de l'historique des pertes dans un fichier CSV.

Ce fichier doit être lancé depuis la racine du projet avec :

    python -m training.train_vae

Exemple pour beta = 1 :

    python -m training.train_vae --beta 1 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vae import VAE
from training.losses import vae_loss


# ================================================================
# CHEMINS DU PROJET
# ================================================================

# __file__ représente le chemin du fichier actuel :
# D:\projet-cvae\training\train_vae.py
#
# parents[1] permet de remonter à :
# D:\projet-cvae
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_DIR = PROJECT_ROOT / "results" / "training_histories"


def set_random_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires pour rendre les expériences reproductibles.

    Sans graine fixe, la séparation train/validation, l'initialisation
    des poids et l'ordre des batchs pourraient changer à chaque exécution.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ces instructions sont utiles lorsqu'une carte graphique CUDA
    # est disponible.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Favorise la reproductibilité sur GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested_device: str) -> torch.device:
    """
    Sélectionne l'appareil utilisé pour les calculs.

    Parameters
    ----------
    requested_device:
        - "auto" : utilise CUDA si disponible, sinon le CPU ;
        - "cpu" : force l'utilisation du processeur ;
        - "cuda" : force l'utilisation d'une carte graphique NVIDIA.
    """

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA a été demandé, mais aucune version CUDA de PyTorch "
            "ou aucune carte graphique compatible n'est disponible."
        )

    return torch.device(requested_device)


def create_dataloaders(
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """
    Charge Fashion-MNIST et crée les DataLoader train et validation.

    Fashion-MNIST fournit officiellement :

        - 60 000 images d'entraînement ;
        - 10 000 images de test.

    Dans ce script, les 60 000 images sont séparées ainsi :

        - 54 000 images pour l'entraînement ;
        - 6 000 images pour la validation.

    Le jeu officiel de test n'est pas chargé ici, car il sera utilisé
    uniquement pendant l'évaluation finale.

    Parameters
    ----------
    batch_size:
        Nombre d'images par batch.

    seed:
        Graine utilisée pour rendre la séparation reproductible.

    num_workers:
        Nombre de processus utilisés pour charger les données.

        Sous Windows, num_workers=0 est la valeur la plus sûre.

    pin_memory:
        Accélère certains transferts vers le GPU lorsqu'il est disponible.

    Returns
    -------
    train_loader:
        DataLoader contenant les 54 000 images d'entraînement.

    validation_loader:
        DataLoader contenant les 6 000 images de validation.
    """

    # ToTensor convertit chaque image en tenseur de forme [1, 28, 28]
    # et transforme les pixels vers l'intervalle [0, 1].
    transform = transforms.ToTensor()

    # Chargement du jeu officiel d'entraînement.
    full_train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    train_size = 54_000
    validation_size = 6_000

    # Vérification de sécurité.
    if train_size + validation_size != len(full_train_dataset):
        raise RuntimeError(
            "La taille attendue de Fashion-MNIST est différente de "
            f"la taille observée : {len(full_train_dataset)}."
        )

    # Le générateur avec une graine fixe garantit que les mêmes images
    # seront toujours placées dans train et validation.
    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        dataset=full_train_dataset,
        lengths=[train_size, validation_size],
        generator=split_generator,
    )

    # Un deuxième générateur contrôle l'ordre aléatoire des batchs
    # d'entraînement.
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    beta: float,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Entraîne le modèle pendant une époque.

    Une époque correspond normalement à un passage complet sur toutes
    les images d'entraînement.

    Le paramètre max_batches sert uniquement aux tests rapides.
    Lors de l'entraînement final, il restera égal à None.

    Returns
    -------
    Un dictionnaire contenant les moyennes par image :

        - total ;
        - reconstruction ;
        - kl.
    """

    # Active le mode entraînement.
    model.train()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    processed_samples = 0

    # Lorsque max_batches est fourni, la barre de progression affiche
    # seulement le nombre de batchs réellement utilisés.
    progress_total = len(dataloader)

    if max_batches is not None:
        progress_total = min(progress_total, max_batches)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Entraînement",
        leave=False,
    )

    for batch_index, (images, _) in progress_bar:
        # Arrête la boucle après le nombre de batchs demandé pour le test.
        if max_batches is not None and batch_index >= max_batches:
            break

        # Le VAE classique n'utilise pas les labels.
        images = images.to(
            device=device,
            non_blocking=True,
        )

        batch_size = images.shape[0]

        # Efface les gradients calculés au batch précédent.
        optimizer.zero_grad(set_to_none=True)

        # Passage des images dans le VAE.
        reconstruction, mu, logvar, _ = model(images)

        # Calcul de la loss ELBO négative.
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        # Calcul des gradients.
        total_loss.backward()

        # Mise à jour des poids du réseau.
        optimizer.step()

        # Les pertes retournées sont des moyennes par image.
        # On les multiplie par batch_size avant de les additionner,
        # afin de calculer ensuite une moyenne exacte sur toute l'époque.
        total_loss_sum += total_loss.item() * batch_size
        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )
        kl_loss_sum += kl_loss.item() * batch_size
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


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Évalue le modèle sur le jeu de validation.

    Contrairement à l'entraînement :

    - aucun gradient n'est calculé ;
    - aucun poids n'est modifié ;
    - les images servent uniquement à mesurer les performances.
    """

    # Active le mode évaluation.
    model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    processed_samples = 0

    progress_total = len(dataloader)

    if max_batches is not None:
        progress_total = min(progress_total, max_batches)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=progress_total,
        desc="Validation",
        leave=False,
    )

    for batch_index, (images, _) in progress_bar:
        if max_batches is not None and batch_index >= max_batches:
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

        total_loss_sum += total_loss.item() * batch_size
        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )
        kl_loss_sum += kl_loss.item() * batch_size
        processed_samples += batch_size

    return calculate_average_metrics(
        total_loss_sum=total_loss_sum,
        reconstruction_loss_sum=reconstruction_loss_sum,
        kl_loss_sum=kl_loss_sum,
        processed_samples=processed_samples,
    )


def calculate_average_metrics(
    total_loss_sum: float,
    reconstruction_loss_sum: float,
    kl_loss_sum: float,
    processed_samples: int,
) -> dict[str, float]:
    """
    Calcule les pertes moyennes par image.

    Cette fonction est utilisée à la fois pour l'entraînement
    et pour la validation.
    """

    if processed_samples <= 0:
        raise RuntimeError(
            "Aucune image n'a été traitée pendant cette époque."
        )

    return {
        "total": total_loss_sum / processed_samples,
        "reconstruction": reconstruction_loss_sum / processed_samples,
        "kl": kl_loss_sum / processed_samples,
    }


def beta_to_tag(beta: float) -> str:
    """
    Transforme beta en texte utilisable dans un nom de fichier.

    Exemples
    --------
    0.1 devient "01"
    1.0 devient "1"
    4.0 devient "4"
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
    Enregistre le meilleur état du modèle.

    Le checkpoint contient suffisamment d'informations pour reconstruire
    le modèle et reprendre ou analyser l'expérience.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": "VAE",
        "epoch": epoch,
        "beta": beta,
        "best_validation_loss": best_validation_loss,
        "configuration": configuration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def save_history_csv(
    history: list[dict[str, float]],
    path: Path,
) -> None:
    """
    Enregistre les pertes de chaque époque dans un fichier CSV.

    Ce fichier sera utilisé plus tard pour tracer les courbes
    d'apprentissage et comparer les différentes valeurs de beta.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

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


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les options disponibles dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description="Entraîner un VAE sur Fashion-MNIST."
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
        help="Nombre d'époques. Valeur par défaut : 20.",
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
        help="Taille de la couche cachée. Valeur par défaut : 256.",
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
            "Sous Windows, conserver 0 au début."
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
        help="Nom personnalisé de l'expérience.",
    )

    # Ces deux paramètres servent uniquement à réaliser un test rapide.
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limite temporaire du nombre de batchs d'entraînement.",
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limite temporaire du nombre de batchs de validation.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Vérifie les arguments fournis dans le terminal.
    """

    if args.beta < 0:
        raise ValueError("beta doit être supérieur ou égal à zéro.")

    if args.epochs <= 0:
        raise ValueError("epochs doit être strictement positif.")

    if args.batch_size <= 0:
        raise ValueError("batch-size doit être strictement positif.")

    if args.latent_dim <= 0:
        raise ValueError("latent-dim doit être strictement positif.")

    if args.hidden_dim <= 0:
        raise ValueError("hidden-dim doit être strictement positif.")

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError("num-workers ne peut pas être négatif.")

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


def main() -> None:
    """
    Point d'entrée principal du script d'entraînement.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_random_seed(args.seed)

    device = select_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader, validation_loader = create_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Création du VAE.
    model = VAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Adam est un optimiseur très utilisé pour les VAE.
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Une expérience limitée en nombre de batchs est considérée
    # comme un test rapide, et non comme un entraînement final.
    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"vae_beta_{beta_to_tag(args.beta)}"

        if is_smoke_test:
            run_name += "_smoke"

    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"
    history_path = HISTORY_DIR / f"{run_name}_history.csv"

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

    print("=" * 65)
    print("ENTRAÎNEMENT DU VAE SUR FASHION-MNIST")
    print("=" * 65)
    print(f"Appareil utilisé            : {device}")
    print(f"Images d'entraînement       : {len(train_loader.dataset)}")
    print(f"Images de validation        : {len(validation_loader.dataset)}")
    print(f"Taille des batchs           : {args.batch_size}")
    print(f"Dimension latente           : {args.latent_dim}")
    print(f"Beta                        : {args.beta}")
    print(f"Nombre d'époques            : {args.epochs}")
    print(f"Taux d'apprentissage        : {args.learning_rate}")
    print(f"Nom de l'expérience         : {run_name}")

    if is_smoke_test:
        print("Mode                        : TEST RAPIDE")

    print("=" * 65)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
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
            "train_total": train_metrics["total"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_kl": train_metrics["kl"],
            "validation_total": validation_metrics["total"],
            "validation_reconstruction": (
                validation_metrics["reconstruction"]
            ),
            "validation_kl": validation_metrics["kl"],
        }

        history.append(history_row)

        print(
            f"Époque {epoch:02d}/{args.epochs:02d} | "
            f"Train total={train_metrics['total']:.4f} | "
            f"Train recon={train_metrics['reconstruction']:.4f} | "
            f"Train KL={train_metrics['kl']:.4f} | "
            f"Val total={validation_metrics['total']:.4f} | "
            f"Val recon={validation_metrics['reconstruction']:.4f} | "
            f"Val KL={validation_metrics['kl']:.4f}"
        )

        # Le meilleur modèle est celui qui possède la plus petite
        # loss totale de validation.
        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]
            best_epoch = epoch

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                beta=args.beta,
                best_validation_loss=best_validation_loss,
                configuration=configuration,
            )

            print(
                f"  -> Nouveau meilleur modèle sauvegardé "
                f"à l'époque {epoch}."
            )

        # L'historique est sauvegardé après chaque époque.
        # En cas d'interruption, les résultats déjà obtenus sont conservés.
        save_history_csv(
            history=history,
            path=history_path,
        )

    print("=" * 65)
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"Meilleure époque            : {best_epoch}")
    print(f"Meilleure loss validation   : {best_validation_loss:.4f}")
    print(f"Checkpoint                  : {checkpoint_path}")
    print(f"Historique CSV              : {history_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()