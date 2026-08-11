"""
Évaluation quantitative et qualitative des VAE et CVAE.

Ce script utilise le jeu officiel de test de Fashion-MNIST, qui contient
10 000 images jamais utilisées pour modifier les poids des modèles.

Il réalise les opérations suivantes :

1. charge les checkpoints entraînés ;
2. reconstruit l'architecture correspondant à chaque checkpoint ;
3. calcule les pertes sur le jeu de test ;
4. sauvegarde un tableau CSV récapitulatif ;
5. sauvegarde des exemples de reconstructions ;
6. sauvegarde des images générées aléatoirement par le VAE ;
7. sauvegarde des images générées par classe avec le CVAE.

Le script doit être exécuté depuis la racine du projet :

    python -m evaluation.evaluate
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.cvae import CVAE
from models.vae import VAE
from training.losses import vae_loss
from training.train_vae import select_device, set_random_seed


# ================================================================
# CHEMINS DU PROJET
# ================================================================

# Le fichier actuel est situé dans :
#
# D:\projet-cvae\evaluation\evaluate.py
#
# parents[1] correspond donc à :
#
# D:\projet-cvae
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

RECONSTRUCTION_DIR = RESULTS_DIR / "reconstructions"
GENERATION_DIR = RESULTS_DIR / "generations"

METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"


# Noms des classes Fashion-MNIST.
CLASS_NAMES = [
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


def create_test_loader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Crée le DataLoader du jeu officiel de test.

    Fashion-MNIST fournit :

        - 60 000 images officielles d'entraînement ;
        - 10 000 images officielles de test.

    Les 10 000 images de test n'ont pas été utilisées pour entraîner
    les modèles ni pour sélectionner les checkpoints.

    Parameters
    ----------
    batch_size:
        Nombre d'images traitées simultanément.

    num_workers:
        Nombre de processus utilisés pour charger les données.

        Sous Windows, la valeur 0 est la plus sûre.

    pin_memory:
        Peut accélérer les transferts lorsque CUDA est utilisé.

    Returns
    -------
    test_loader:
        DataLoader contenant les 10 000 images de test.
    """

    # Convertit les images en tenseurs de forme [1, 28, 28].
    #
    # Les pixels sont transformés vers l'intervalle [0, 1].
    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    if len(test_dataset) != 10_000:
        raise RuntimeError(
            "Le jeu de test Fashion-MNIST devrait contenir "
            f"10 000 images, mais {len(test_dataset)} ont été trouvées."
        )

    # shuffle=False permet de conserver toujours le même ordre.
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return test_loader


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Charge un checkpoint et reconstruit le modèle correspondant.

    Le checkpoint contient une clé "model_type" indiquant s'il s'agit
    d'un VAE ou d'un CVAE.

    Parameters
    ----------
    checkpoint_path:
        Chemin du fichier .pt.

    device:
        Appareil sur lequel le modèle sera chargé.

    Returns
    -------
    model:
        VAE ou CVAE avec ses poids entraînés.

    checkpoint:
        Dictionnaire complet chargé depuis le fichier .pt.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "model_type",
        "epoch",
        "beta",
        "configuration",
        "model_state_dict",
    }

    missing_keys = required_keys.difference(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            "Le checkpoint ne contient pas toutes les informations "
            f"attendues. Clés absentes : {sorted(missing_keys)}."
        )

    configuration = checkpoint["configuration"]
    model_type = checkpoint["model_type"]

    if model_type == "VAE":
        model = VAE(
            latent_dim=configuration["latent_dim"],
            hidden_dim=configuration["hidden_dim"],
        )

    elif model_type == "CVAE":
        model = CVAE(
            latent_dim=configuration["latent_dim"],
            hidden_dim=configuration["hidden_dim"],
            num_classes=configuration["num_classes"],
        )

    else:
        raise ValueError(
            f"Type de modèle inconnu dans le checkpoint : {model_type}."
        )

    # Recharge les poids appris.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Place le modèle sur le CPU ou le GPU sélectionné.
    model = model.to(device)

    # Passe le modèle en mode évaluation.
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Calcule les pertes moyennes sur le jeu de test.

    Parameters
    ----------
    model:
        Modèle VAE ou CVAE.

    model_type:
        Chaîne "VAE" ou "CVAE".

    dataloader:
        DataLoader du jeu de test.

    device:
        Appareil utilisé pour les calculs.

    beta:
        Poids du terme KL enregistré dans le checkpoint.

    max_batches:
        Limite optionnelle utilisée uniquement pour un test rapide.

    Returns
    -------
    metrics:
        Dictionnaire contenant :

        - total ;
        - reconstruction ;
        - kl ;
        - processed_samples.
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
        desc=f"Évaluation {model_type}",
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

        # Le VAE reçoit seulement les images.
        if model_type == "VAE":
            reconstruction, mu, logvar, _ = model(images)

        # Le CVAE reçoit les images et leurs labels.
        elif model_type == "CVAE":
            reconstruction, mu, logvar, _ = model(
                images,
                labels,
            )

        else:
            raise ValueError(
                f"Type de modèle non pris en charge : {model_type}."
            )

        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction=reconstruction,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        # Conversion des moyennes de batch en sommes.
        total_loss_sum += total_loss.item() * batch_size

        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )

        kl_loss_sum += kl_loss.item() * batch_size

        processed_samples += batch_size

    if processed_samples == 0:
        raise RuntimeError(
            "Aucune image n'a été traitée pendant l'évaluation."
        )

    return {
        "total": total_loss_sum / processed_samples,
        "reconstruction": (
            reconstruction_loss_sum / processed_samples
        ),
        "kl": kl_loss_sum / processed_samples,
        "processed_samples": processed_samples,
    }


@torch.no_grad()
def save_reconstruction_examples(
    model: nn.Module,
    model_type: str,
    dataloader: DataLoader,
    device: torch.device,
    output_path: Path,
    num_images: int,
) -> None:
    """
    Sauvegarde des images originales et leurs reconstructions.

    La grille produite contient :

        première ligne  : images originales ;
        deuxième ligne : images reconstruites.

    Parameters
    ----------
    model:
        Modèle à utiliser.

    model_type:
        "VAE" ou "CVAE".

    dataloader:
        DataLoader du jeu de test.

    device:
        Appareil de calcul.

    output_path:
        Chemin du fichier PNG à produire.

    num_images:
        Nombre d'exemples à afficher.
    """

    if num_images <= 0:
        raise ValueError(
            "num_images doit être strictement positif."
        )

    model.eval()

    # Récupère le premier batch du jeu de test.
    images, labels = next(iter(dataloader))

    num_images = min(
        num_images,
        images.shape[0],
    )

    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)

    if model_type == "VAE":
        reconstructions, _, _, _ = model(images)

    elif model_type == "CVAE":
        reconstructions, _, _, _ = model(
            images,
            labels,
        )

    else:
        raise ValueError(
            f"Type de modèle non pris en charge : {model_type}."
        )

    # Concatène d'abord les originales, puis les reconstructions.
    #
    # Avec nrow=num_images, save_image produira deux lignes.
    comparison = torch.cat(
        tensors=(images, reconstructions),
        dim=0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        tensor=comparison.cpu(),
        fp=str(output_path),
        nrow=num_images,
        padding=2,
    )


@torch.no_grad()
def save_generation_examples(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    output_path: Path,
    samples_per_class: int,
) -> None:
    """
    Sauvegarde des images nouvellement générées.

    Pour le VAE :
        les images sont générées sans classe imposée.

    Pour le CVAE :
        chaque ligne correspond à une classe Fashion-MNIST.

    Parameters
    ----------
    samples_per_class:
        Nombre d'images produites pour chaque classe.

        Pour le VAE, le nombre total produit est :

            10 × samples_per_class
    """

    if samples_per_class <= 0:
        raise ValueError(
            "samples_per_class doit être strictement positif."
        )

    model.eval()

    num_classes = len(CLASS_NAMES)

    if model_type == "VAE":
        # Le VAE ne reçoit aucun label.
        #
        # On produit le même nombre total d'images que pour le CVAE
        # afin de faciliter la comparaison visuelle.
        num_samples = num_classes * samples_per_class

        generated_images = model.sample(
            num_samples=num_samples,
            device=device,
        )

        # Dix images par ligne lorsque samples_per_class vaut 10.
        nrow = samples_per_class

    elif model_type == "CVAE":
        # Exemple avec samples_per_class=5 :
        #
        # [0, 0, 0, 0, 0,
        #  1, 1, 1, 1, 1,
        #  ...
        #  9, 9, 9, 9, 9]
        #
        # Chaque ligne de la grille correspondra à une classe.
        labels = torch.arange(
            start=0,
            end=num_classes,
            device=device,
            dtype=torch.long,
        )

        labels = labels.repeat_interleave(
            samples_per_class
        )

        generated_images = model.sample(
            labels=labels,
            device=device,
        )

        nrow = samples_per_class

    else:
        raise ValueError(
            f"Type de modèle non pris en charge : {model_type}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_image(
        tensor=generated_images.cpu(),
        fp=str(output_path),
        nrow=nrow,
        padding=2,
    )


def save_metrics_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """
    Enregistre les résultats quantitatifs dans un fichier CSV.
    """

    if not rows:
        raise ValueError(
            "Aucun résultat n'est disponible pour créer le CSV."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "checkpoint",
        "model_type",
        "beta",
        "best_epoch",
        "best_validation_loss",
        "test_total",
        "test_reconstruction",
        "test_kl",
        "processed_test_images",
    ]

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
        writer.writerows(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments utilisables dans le terminal.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Évaluer les VAE et CVAE entraînés sur Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Taille des batchs de test. Valeur par défaut : 128.",
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
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire. Valeur par défaut : 42.",
    )

    parser.add_argument(
        "--num-reconstruction-images",
        type=int,
        default=10,
        help=(
            "Nombre d'images utilisées dans les grilles "
            "de reconstruction."
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help=(
            "Nombre d'images générées pour chacune des dix classes."
        ),
    )

    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help=(
            "Limite temporaire du nombre de batchs de test. "
            "À utiliser uniquement pour une vérification rapide."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/vae_beta_01.pt",
            "checkpoints/vae_beta_1.pt",
            "checkpoints/vae_beta_4.pt",
            "checkpoints/cvae_beta_01.pt",
            "checkpoints/cvae_beta_1.pt",
            "checkpoints/cvae_beta_4.pt",
        ],
        help="Liste des checkpoints à évaluer.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Vérifie les paramètres reçus depuis le terminal.
    """

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
        )

    if args.num_reconstruction_images <= 0:
        raise ValueError(
            "num-reconstruction-images doit être positif."
        )

    if args.samples_per_class <= 0:
        raise ValueError(
            "samples-per-class doit être positif."
        )

    if (
        args.max_test_batches is not None
        and args.max_test_batches <= 0
    ):
        raise ValueError(
            "max-test-batches doit être strictement positif."
        )


def resolve_checkpoint_path(path_text: str) -> Path:
    """
    Convertit un chemin fourni dans le terminal en chemin absolu.

    Un chemin relatif est interprété depuis la racine du projet.
    """

    path = Path(path_text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def main() -> None:
    """
    Point d'entrée principal du script d'évaluation.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_random_seed(args.seed)

    device = select_device(args.device)
    pin_memory = device.type == "cuda"

    test_loader = create_test_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("=" * 70)
    print("ÉVALUATION DES MODÈLES SUR FASHION-MNIST")
    print("=" * 70)
    print(f"Appareil utilisé          : {device}")
    print(f"Images officielles de test : {len(test_loader.dataset)}")
    print(f"Taille des batchs         : {args.batch_size}")

    if args.max_test_batches is not None:
        print(
            "Mode                      : TEST RAPIDE "
            f"({args.max_test_batches} batchs)"
        )

    print("=" * 70)

    results: list[dict] = []

    for checkpoint_text in args.checkpoints:
        checkpoint_path = resolve_checkpoint_path(
            checkpoint_text
        )

        print()
        print(f"Chargement : {checkpoint_path.name}")

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        model_type = checkpoint["model_type"]
        beta = float(checkpoint["beta"])

        metrics = evaluate_model(
            model=model,
            model_type=model_type,
            dataloader=test_loader,
            device=device,
            beta=beta,
            max_batches=args.max_test_batches,
        )

        reconstruction_path = (
            RECONSTRUCTION_DIR
            / f"{checkpoint_path.stem}_reconstructions.png"
        )

        generation_path = (
            GENERATION_DIR
            / f"{checkpoint_path.stem}_generations.png"
        )

        save_reconstruction_examples(
            model=model,
            model_type=model_type,
            dataloader=test_loader,
            device=device,
            output_path=reconstruction_path,
            num_images=args.num_reconstruction_images,
        )

        save_generation_examples(
            model=model,
            model_type=model_type,
            device=device,
            output_path=generation_path,
            samples_per_class=args.samples_per_class,
        )

        result_row = {
            "checkpoint": checkpoint_path.name,
            "model_type": model_type,
            "beta": beta,
            "best_epoch": checkpoint["epoch"],
            "best_validation_loss": (
                checkpoint["best_validation_loss"]
            ),
            "test_total": metrics["total"],
            "test_reconstruction": metrics["reconstruction"],
            "test_kl": metrics["kl"],
            "processed_test_images": (
                metrics["processed_samples"]
            ),
        }

        results.append(result_row)

        print(f"Modèle                    : {model_type}")
        print(f"Beta                      : {beta}")
        print(f"Époque du checkpoint      : {checkpoint['epoch']}")
        print(f"Images testées            : {metrics['processed_samples']}")
        print(f"Test total                : {metrics['total']:.4f}")
        print(
            "Test reconstruction       : "
            f"{metrics['reconstruction']:.4f}"
        )
        print(f"Test KL                   : {metrics['kl']:.4f}")
        print(f"Reconstructions sauvegardées : {reconstruction_path}")
        print(f"Générations sauvegardées     : {generation_path}")

        # Libère explicitement le modèle avant de charger le suivant.
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_metrics_csv(
        rows=results,
        output_path=METRICS_PATH,
    )

    print()
    print("=" * 70)
    print("ÉVALUATION TERMINÉE")
    print(f"Tableau des métriques : {METRICS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()