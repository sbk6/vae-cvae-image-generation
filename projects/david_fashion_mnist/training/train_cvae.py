"""
Entraînement du CVAE sur Fashion-MNIST.

Le CVAE est un VAE conditionnel. Il reçoit deux informations :

    - une image ;
    - son label de classe.

Les labels Fashion-MNIST sont :

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

Ce script réalise les opérations suivantes :

1. charge Fashion-MNIST ;
2. sépare les 60 000 images officielles d'entraînement :
       - 54 000 images pour l'entraînement ;
       - 6 000 images pour la validation ;
3. entraîne le CVAE avec les images et leurs labels ;
4. mesure les pertes sur le jeu de validation ;
5. sauvegarde le meilleur modèle ;
6. sauvegarde l'historique dans un fichier CSV.

Le script doit être lancé depuis la racine du projet :

    python -m training.train_cvae

Exemple :

    python -m training.train_cvae --beta 1 --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cvae import CVAE
from training.losses import vae_loss

# Plusieurs fonctions générales sont déjà définies dans train_vae.py.
#
# Nous les réutilisons afin d'éviter de recopier inutilement :
#
# - la création des DataLoader ;
# - la fixation des graines aléatoires ;
# - la sélection CPU/GPU ;
# - le calcul des moyennes ;
# - la sauvegarde de l'historique CSV.
from training.train_vae import (
    beta_to_tag,
    calculate_average_metrics,
    create_dataloaders,
    save_history_csv,
    select_device,
    set_random_seed,
)


# ================================================================
# CHEMINS DU PROJET
# ================================================================

# Le fichier actuel se trouve dans :
#
# D:\projet-cvae\training\train_cvae.py
#
# parents[1] permet donc de récupérer :
#
# D:\projet-cvae
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_DIR = PROJECT_ROOT / "results" / "training_histories"


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

    Parameters
    ----------
    model:
        Le modèle CVAE à entraîner.

    dataloader:
        DataLoader contenant les images et les labels.

    optimizer:
        Optimiseur chargé de modifier les poids du modèle.

    device:
        Appareil de calcul utilisé : CPU ou GPU.

    beta:
        Poids appliqué au terme KL.

    max_batches:
        Limite optionnelle du nombre de batchs.

        Cette option sert uniquement pour les tests rapides.

    Returns
    -------
    metrics:
        Dictionnaire contenant les pertes moyennes par image :

        - total ;
        - reconstruction ;
        - kl.
    """

    # Active le mode entraînement.
    #
    # Cette instruction est importante pour les couches dont le
    # comportement diffère entre entraînement et évaluation.
    model.train()

    # Sommes utilisées pour calculer les moyennes sur toute l'époque.
    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0

    # Nombre total d'images réellement traitées.
    processed_samples = 0

    # Nombre de batchs affiché dans la barre de progression.
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
        # Dans un test rapide, on s'arrête après le nombre demandé.
        if max_batches is not None and batch_index >= max_batches:
            break

        # Envoie les images sur le CPU ou le GPU sélectionné.
        images = images.to(
            device=device,
            non_blocking=True,
        )

        # Les labels doivent être placés sur le même appareil.
        #
        # Fashion-MNIST fournit déjà des labels de type torch.int64,
        # qui est le type attendu par notre CVAE.
        labels = labels.to(
            device=device,
            non_blocking=True,
        )

        batch_size = images.shape[0]

        # Efface les gradients calculés lors du batch précédent.
        optimizer.zero_grad(set_to_none=True)

        # Passage dans le CVAE.
        #
        # Le label est utilisé :
        #
        # - par l'encodeur ;
        # - par le décodeur.
        reconstruction, mu, logvar, _ = model(
            images,
            labels,
        )

        # La loss est identique à celle du VAE classique.
        #
        # Le conditionnement est pris en charge par l'architecture
        # et non directement par la fonction de perte.
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        # Calcule les gradients de la loss par rapport aux poids.
        total_loss.backward()

        # Met à jour les poids du CVAE.
        optimizer.step()

        # Les pertes renvoyées par vae_loss sont déjà des moyennes
        # par image dans le batch.
        #
        # On les multiplie par le nombre d'images pour pouvoir calculer
        # ensuite une moyenne exacte sur toute l'époque.
        total_loss_sum += total_loss.item() * batch_size

        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )

        kl_loss_sum += kl_loss.item() * batch_size

        processed_samples += batch_size

        # Affichage des pertes du batch actuel.
        progress_bar.set_postfix(
            total=f"{total_loss.item():.2f}",
            reconstruction=f"{reconstruction_loss.item():.2f}",
            kl=f"{kl_loss.item():.2f}",
        )

    # Calcule les trois pertes moyennes par image.
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
    Évalue le CVAE sur le jeu de validation.

    Pendant la validation :

    - aucun gradient n'est calculé ;
    - aucun poids n'est modifié ;
    - le modèle est seulement mesuré.

    Le décorateur @torch.no_grad() réduit la consommation de mémoire
    et accélère les calculs.
    """

    # Active le mode évaluation.
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
        if max_batches is not None and batch_index >= max_batches:
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

        # Le CVAE reçoit toujours les labels pendant la validation.
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

    Le checkpoint contient :

    - le type du modèle ;
    - le numéro de l'époque ;
    - la valeur de beta ;
    - la meilleure perte de validation ;
    - les hyperparamètres ;
    - les poids du modèle ;
    - l'état de l'optimiseur.
    """

    # Crée le dossier checkpoints s'il n'existe pas.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "CVAE",
        "epoch": epoch,
        "beta": beta,
        "best_validation_loss": best_validation_loss,
        "configuration": configuration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # torch.save sérialise le dictionnaire dans un fichier .pt.
    torch.save(
        checkpoint,
        path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments disponibles dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description="Entraîner un CVAE sur Fashion-MNIST."
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
        "--num-classes",
        type=int,
        default=10,
        help="Nombre de classes. Fashion-MNIST en possède 10.",
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
            "Nombre de processus utilisés pour charger les données. "
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

    # Ces deux arguments servent seulement aux tests rapides.
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
    Vérifie la validité des arguments fournis par l'utilisateur.
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

    if args.num_classes <= 1:
        raise ValueError(
            "num-classes doit être supérieur à un."
        )

    if args.num_classes != 10:
        raise ValueError(
            "Fashion-MNIST contient exactement 10 classes."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "learning-rate doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
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


def main() -> None:
    """
    Point d'entrée principal de l'entraînement du CVAE.
    """

    # Lecture des arguments du terminal.
    parser = build_argument_parser()
    args = parser.parse_args()

    # Vérifie les valeurs reçues.
    validate_arguments(args)

    # Rend l'expérience reproductible.
    set_random_seed(args.seed)

    # Sélectionne automatiquement le CPU dans votre environnement actuel.
    device = select_device(args.device)

    # pin_memory est surtout utile avec CUDA.
    pin_memory = device.type == "cuda"

    # Réutilise exactement la même séparation train/validation
    # que celle utilisée pour le VAE.
    #
    # C'est important pour comparer équitablement VAE et CVAE.
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

    # Création de l'optimiseur Adam.
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Une expérience limitée en nombre de batchs est considérée
    # comme un test rapide.
    is_smoke_test = (
        args.max_train_batches is not None
        or args.max_val_batches is not None
    )

    # Construction du nom de l'expérience.
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"cvae_beta_{beta_to_tag(args.beta)}"

        if is_smoke_test:
            run_name += "_smoke"

    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"

    history_path = (
        HISTORY_DIR / f"{run_name}_history.csv"
    )

    # Configuration enregistrée dans le checkpoint.
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

    print("=" * 65)
    print("ENTRAÎNEMENT DU CVAE SUR FASHION-MNIST")
    print("=" * 65)
    print(f"Appareil utilisé            : {device}")
    print(f"Images d'entraînement       : {len(train_loader.dataset)}")
    print(f"Images de validation        : {len(validation_loader.dataset)}")
    print(f"Taille des batchs           : {args.batch_size}")
    print(f"Dimension latente           : {args.latent_dim}")
    print(f"Nombre de classes           : {args.num_classes}")
    print(f"Beta                        : {args.beta}")
    print(f"Nombre d'époques            : {args.epochs}")
    print(f"Taux d'apprentissage        : {args.learning_rate}")
    print(f"Nom de l'expérience         : {run_name}")

    if is_smoke_test:
        print("Mode                        : TEST RAPIDE")

    print("=" * 65)

    # Historique contenant une ligne par époque.
    history: list[dict[str, float]] = []

    # La meilleure loss commence à l'infini.
    best_validation_loss = float("inf")

    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Entraînement pendant une époque.
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=args.beta,
            max_batches=args.max_train_batches,
        )

        # Évaluation sur la validation.
        validation_metrics = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            beta=args.beta,
            max_batches=args.max_val_batches,
        )

        # Ligne enregistrée dans le fichier CSV.
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

        # Sauvegarde le modèle seulement lorsque la validation s'améliore.
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

        # Sauvegarde l'historique après chaque époque.
        save_history_csv(
            history=history,
            path=history_path,
        )

    print("=" * 65)
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"Meilleure époque            : {best_epoch}")
    print(
        f"Meilleure loss validation   : "
        f"{best_validation_loss:.4f}"
    )
    print(f"Checkpoint                  : {checkpoint_path}")
    print(f"Historique CSV              : {history_path}")
    print("=" * 65)


# Ce bloc est exécuté uniquement lorsque le module est lancé directement.
if __name__ == "__main__":
    main()