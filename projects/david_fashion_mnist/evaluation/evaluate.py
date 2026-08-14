"""
Évaluation quantitative et qualitative des VAE et CVAE.

Ce script peut évaluer les modèles sur deux ensembles :

1. validation :
       - 6 000 images issues des 60 000 images officielles
         d'entraînement de Fashion-MNIST ;
       - utilisé pour comparer les modèles et sélectionner
         les hyperparamètres ;

2. test :
       - 10 000 images du jeu officiel de test ;
       - réservé à l'évaluation finale après sélection du modèle.

Pour éviter toute fuite méthodologique pendant la sélection des modèles,
le mode par défaut est :

    --split validation

Les métriques calculées sont :

    - loss totale du beta-VAE / beta-CVAE, avec un échantillon
      Monte-Carlo z ~ q(z|x), comme pendant l'entraînement ;
    - reconstruction BCE avec ce même échantillon ;
    - divergence KL ;
    - reconstruction BCE déterministe avec z = mu ;
    - MSE déterministe avec z = mu ;
    - SSIM déterministe avec z = mu.

Pour les métriques de fidélité visuelle (BCE déterministe, MSE et SSIM),
la reconstruction utilise z = mu. Cela supprime le bruit dû à
l'échantillonnage latent et rend les comparaisons entre checkpoints
reproductibles.

Le script sauvegarde également :

    - des exemples de reconstructions ;
    - des générations aléatoires pour le VAE ;
    - des générations conditionnelles par classe pour le CVAE ;
    - un tableau CSV récapitulatif.

Exemple pour la validation :

    python -m evaluation.evaluate --split validation

Exemple pour le test final :

    python -m evaluation.evaluate --split test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.cvae import CVAE
from models.vae import VAE
from training.losses import vae_loss
from training.train_vae import (
    create_dataloaders,
    select_device,
    set_random_seed,
)


# ================================================================
# CHEMINS DU PROJET
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

RECONSTRUCTION_DIR = (
    RESULTS_DIR
    / "reconstructions"
)

GENERATION_DIR = (
    RESULTS_DIR
    / "generations"
)

VALIDATION_METRICS_PATH = (
    RESULTS_DIR
    / "validation_metrics_final.csv"
)

TEST_METRICS_PATH = (
    RESULTS_DIR
    / "test_metrics_final.csv"
)


# ================================================================
# CLASSES FASHION-MNIST
# ================================================================

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


# ================================================================
# DONNÉES DE VALIDATION
# ================================================================


def create_validation_loader(
    batch_size: int,
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Recrée exactement le jeu de validation utilisé à l'entraînement.

    Les 60 000 images officielles d'entraînement sont séparées en :

        - 54 000 images d'entraînement ;
        - 6 000 images de validation.

    Le split est contrôlé par split_seed.

    Pour les expériences principales du projet :

        split_seed = 42
    """

    _, validation_loader = create_dataloaders(
        batch_size=batch_size,
        seed=split_seed,
        split_seed=split_seed,
        loader_seed=split_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if len(validation_loader.dataset) != 6_000:
        raise RuntimeError(
            "Le jeu de validation devrait contenir "
            "6 000 images."
        )

    return validation_loader


# ================================================================
# DONNÉES DE TEST
# ================================================================


def create_test_loader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Crée le DataLoader du jeu officiel de test.

    Fashion-MNIST contient 10 000 images officielles de test.

    Ce jeu doit être utilisé uniquement pour la mesure finale,
    après sélection des hyperparamètres.
    """

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
            f"10 000 images, mais {len(test_dataset)} "
            "ont été trouvées."
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return test_loader


# ================================================================
# CHOIX VALIDATION / TEST
# ================================================================


def create_evaluation_loader(
    split: str,
    batch_size: int,
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Retourne le DataLoader correspondant au split demandé.
    """

    if split == "validation":

        return create_validation_loader(
            batch_size=batch_size,
            split_seed=split_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if split == "test":

        return create_test_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(
        f"Split non pris en charge : {split}"
    )


# ================================================================
# CHARGEMENT DES MODÈLES
# ================================================================


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Charge un checkpoint VAE ou CVAE et reconstruit son architecture.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint introuvable : "
            f"{checkpoint_path}"
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

    missing_keys = (
        required_keys
        .difference(
            checkpoint.keys()
        )
    )

    if missing_keys:

        raise KeyError(
            "Le checkpoint ne contient pas toutes "
            "les informations attendues. "
            f"Clés absentes : "
            f"{sorted(missing_keys)}."
        )

    configuration = (
        checkpoint["configuration"]
    )

    model_type = (
        checkpoint["model_type"]
    )

    if model_type == "VAE":

        model = VAE(
            latent_dim=(
                configuration["latent_dim"]
            ),
            hidden_dim=(
                configuration["hidden_dim"]
            ),
        )

    elif model_type == "CVAE":

        model = CVAE(
            latent_dim=(
                configuration["latent_dim"]
            ),
            hidden_dim=(
                configuration["hidden_dim"]
            ),
            num_classes=(
                configuration["num_classes"]
            ),
        )

    else:

        raise ValueError(
            "Type de modèle inconnu dans "
            f"le checkpoint : {model_type}."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    model.eval()

    return model, checkpoint


# ================================================================
# SSIM
# ================================================================


def create_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Crée une fenêtre gaussienne 2D pour le calcul du SSIM.

    Les paramètres utilisés correspondent aux valeurs courantes
    de la formulation classique du SSIM :

        window_size = 11
        sigma       = 1.5
    """

    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype,
    )

    coordinates = (
        coordinates
        - window_size // 2
    )

    gaussian_1d = torch.exp(
        -(
            coordinates ** 2
        )
        / (
            2.0
            * sigma ** 2
        )
    )

    gaussian_1d = (
        gaussian_1d
        / gaussian_1d.sum()
    )

    gaussian_2d = torch.outer(
        gaussian_1d,
        gaussian_1d,
    )

    window = gaussian_2d.view(
        1,
        1,
        window_size,
        window_size,
    )

    window = window.expand(
        channels,
        1,
        window_size,
        window_size,
    )

    return window


def calculate_ssim_per_image(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Calcule le SSIM moyen pour chaque image d'un batch.

    Parameters
    ----------
    reconstruction:
        Tenseur [N, C, H, W].

    target:
        Tenseur [N, C, H, W].

    window_size:
        Taille de la fenêtre gaussienne.

    sigma:
        Écart-type de la fenêtre gaussienne.

    data_range:
        Étendue des valeurs de pixels.

        Fashion-MNIST est représenté dans [0, 1],
        donc data_range = 1.0.

    Returns
    -------
    Tensor
        Un score SSIM par image.

    Notes
    -----
    SSIM proche de 1 :
        forte similarité structurelle.

    SSIM plus faible :
        reconstruction moins fidèle structurellement.
    """

    if reconstruction.shape != target.shape:

        raise ValueError(
            "reconstruction et target doivent "
            "avoir exactement la même forme."
        )

    if reconstruction.ndim != 4:

        raise ValueError(
            "Les tenseurs SSIM doivent avoir "
            "la forme [N, C, H, W]."
        )

    _, channels, height, width = (
        reconstruction.shape
    )

    if (
        height < window_size
        or width < window_size
    ):

        raise ValueError(
            "Les images sont trop petites pour "
            f"une fenêtre SSIM de taille "
            f"{window_size}."
        )

    window = create_gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=reconstruction.device,
        dtype=reconstruction.dtype,
    )

    # Moyennes locales.
    mu_reconstruction = F.conv2d(
        reconstruction,
        window,
        groups=channels,
    )

    mu_target = F.conv2d(
        target,
        window,
        groups=channels,
    )

    mu_reconstruction_squared = (
        mu_reconstruction ** 2
    )

    mu_target_squared = (
        mu_target ** 2
    )

    mu_product = (
        mu_reconstruction
        * mu_target
    )

    # Variances et covariance locales.
    variance_reconstruction = (
        F.conv2d(
            reconstruction
            * reconstruction,
            window,
            groups=channels,
        )
        - mu_reconstruction_squared
    )

    variance_target = (
        F.conv2d(
            target
            * target,
            window,
            groups=channels,
        )
        - mu_target_squared
    )

    covariance = (
        F.conv2d(
            reconstruction
            * target,
            window,
            groups=channels,
        )
        - mu_product
    )

    # Constantes de stabilisation de la formule SSIM.
    c1 = (
        0.01
        * data_range
    ) ** 2

    c2 = (
        0.03
        * data_range
    ) ** 2

    numerator = (
        (
            2.0
            * mu_product
            + c1
        )
        *
        (
            2.0
            * covariance
            + c2
        )
    )

    denominator = (
        (
            mu_reconstruction_squared
            + mu_target_squared
            + c1
        )
        *
        (
            variance_reconstruction
            + variance_target
            + c2
        )
    )

    ssim_map = (
        numerator
        / denominator
    )

    # Moyenne spatiale et moyenne sur les canaux,
    # tout en conservant une valeur par image.
    ssim_per_image = ssim_map.mean(
        dim=(1, 2, 3)
    )

    return ssim_per_image


# ================================================================
# RECONSTRUCTION DÉTERMINISTE
# ================================================================


def decode_from_mu(
    model: nn.Module,
    model_type: str,
    mu: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruit les images à partir de la moyenne latente mu.

    Cette reconstruction est déterministe :

        z = mu

    Elle est utilisée pour les métriques de fidélité visuelle
    (BCE déterministe, MSE et SSIM), ainsi que pour les exemples
    de reconstruction sauvegardés.

    La loss ELBO reste calculée avec un échantillon z ~ q(z|x),
    afin de rester cohérente avec l'objectif d'entraînement.
    """

    if model_type == "VAE":

        return model.decode(
            mu
        )

    if model_type == "CVAE":

        return model.decode(
            mu,
            labels,
        )

    raise ValueError(
        "Type de modèle non pris en charge : "
        f"{model_type}."
    )


# ================================================================
# ÉVALUATION QUANTITATIVE
# ================================================================


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
    Calcule les métriques moyennes sur validation ou test.

    Métriques retournées :

        - total ELBO estimé avec un échantillon z ~ q(z|x) ;
        - reconstruction BCE avec ce même échantillon ;
        - KL ;
        - reconstruction BCE déterministe avec z = mu ;
        - MSE déterministe avec z = mu ;
        - SSIM déterministe avec z = mu ;
        - nombre d'images traitées.

    Pour rendre la comparaison des métriques de reconstruction
    reproductible, BCE déterministe, MSE et SSIM sont calculés
    à partir de z = mu.
    """

    model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0

    deterministic_reconstruction_bce_sum = 0.0

    mse_sum = 0.0
    ssim_sum = 0.0

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
        desc=f"Évaluation {model_type}",
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

        batch_size = (
            images.shape[0]
        )

        # --------------------------------------------------------
        # RECONSTRUCTION
        # --------------------------------------------------------

        if model_type == "VAE":

            (
                sampled_reconstruction,
                mu,
                logvar,
                _,
            ) = model(
                images
            )

        elif model_type == "CVAE":

            (
                sampled_reconstruction,
                mu,
                logvar,
                _,
            ) = model(
                images,
                labels,
            )

        else:

            raise ValueError(
                "Type de modèle non pris en charge : "
                f"{model_type}."
            )

        # --------------------------------------------------------
        # BCE + KL + TOTAL
        # --------------------------------------------------------

        (
            total_loss,
            reconstruction_loss,
            kl_loss,
        ) = vae_loss(
            reconstruction=sampled_reconstruction,
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

        # --------------------------------------------------------
        # RECONSTRUCTION DÉTERMINISTE : z = mu
        # --------------------------------------------------------

        deterministic_reconstruction = decode_from_mu(
            model=model,
            model_type=model_type,
            mu=mu,
            labels=labels,
        )

        # BCE déterministe, somme des pixels par image puis
        # accumulation sur l'ensemble évalué.
        deterministic_bce_per_image = (
            F.binary_cross_entropy(
                deterministic_reconstruction,
                images,
                reduction="none",
            )
            .sum(
                dim=(1, 2, 3)
            )
        )

        deterministic_reconstruction_bce_sum += (
            deterministic_bce_per_image
            .sum()
            .item()
        )

        # --------------------------------------------------------
        # MSE DÉTERMINISTE
        # --------------------------------------------------------

        # MSE moyen par pixel pour chacune des images.
        mse_per_image = (
            (
                deterministic_reconstruction
                - images
            )
            ** 2
        ).mean(
            dim=(1, 2, 3)
        )

        mse_sum += (
            mse_per_image
            .sum()
            .item()
        )

        # --------------------------------------------------------
        # SSIM
        # --------------------------------------------------------

        ssim_per_image = (
            calculate_ssim_per_image(
                reconstruction=(
                    deterministic_reconstruction
                ),
                target=images,
                window_size=11,
                sigma=1.5,
                data_range=1.0,
            )
        )

        ssim_sum += (
            ssim_per_image
            .sum()
            .item()
        )

        processed_samples += (
            batch_size
        )

    if processed_samples == 0:

        raise RuntimeError(
            "Aucune image n'a été traitée "
            "pendant l'évaluation."
        )

    return {
        "total": (
            total_loss_sum
            / processed_samples
        ),

        "reconstruction": (
            reconstruction_loss_sum
            / processed_samples
        ),

        "kl": (
            kl_loss_sum
            / processed_samples
        ),

        "deterministic_reconstruction_bce": (
            deterministic_reconstruction_bce_sum
            / processed_samples
        ),

        "mse": (
            mse_sum
            / processed_samples
        ),

        "ssim": (
            ssim_sum
            / processed_samples
        ),

        "processed_samples": (
            processed_samples
        ),
    }


# ================================================================
# EXEMPLES DE RECONSTRUCTION
# ================================================================


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
    Sauvegarde les images originales et leurs reconstructions.

    Première ligne :
        images originales.

    Deuxième ligne :
        images reconstruites avec z = mu.

    Les reconstructions sont donc déterministes.
    """

    if num_images <= 0:

        raise ValueError(
            "num_images doit être "
            "strictement positif."
        )

    model.eval()

    images, labels = next(
        iter(dataloader)
    )

    num_images = min(
        num_images,
        images.shape[0],
    )

    images = (
        images[:num_images]
        .to(device)
    )

    labels = (
        labels[:num_images]
        .to(device)
    )

    if model_type == "VAE":

        _, mu, _, _ = model(
            images
        )

    elif model_type == "CVAE":

        _, mu, _, _ = model(
            images,
            labels,
        )

    else:

        raise ValueError(
            "Type de modèle non pris en charge : "
            f"{model_type}."
        )

    reconstructions = decode_from_mu(
        model=model,
        model_type=model_type,
        mu=mu,
        labels=labels,
    )

    comparison = torch.cat(
        tensors=(
            images,
            reconstructions,
        ),
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


# ================================================================
# EXEMPLES DE GÉNÉRATION
# ================================================================


@torch.no_grad()
def save_generation_examples(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    output_path: Path,
    samples_per_class: int,
) -> None:
    """
    Sauvegarde des images générées.

    VAE :
        génération non conditionnelle.

    CVAE :
        une ligne par classe Fashion-MNIST.
    """

    if samples_per_class <= 0:

        raise ValueError(
            "samples_per_class doit être "
            "strictement positif."
        )

    model.eval()

    num_classes = len(
        CLASS_NAMES
    )

    if model_type == "VAE":

        num_samples = (
            num_classes
            * samples_per_class
        )

        generated_images = model.sample(
            num_samples=num_samples,
            device=device,
        )

        nrow = samples_per_class

    elif model_type == "CVAE":

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
            "Type de modèle non pris en charge : "
            f"{model_type}."
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


# ================================================================
# CSV
# ================================================================


def save_metrics_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    """
    Enregistre les résultats quantitatifs dans un fichier CSV.
    """

    if not rows:

        raise ValueError(
            "Aucun résultat n'est disponible "
            "pour créer le CSV."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "evaluation_split",
        "checkpoint",
        "model_type",
        "beta",
        "training_seed",
        "split_seed_protocol",
        "checkpoint_split_seed",
        "evaluation_seed",
        "best_epoch",
        "best_validation_loss",
        "evaluation_total_sampled",
        "evaluation_reconstruction_bce_sampled",
        "evaluation_kl",
        "evaluation_reconstruction_bce_deterministic",
        "evaluation_mse_deterministic",
        "evaluation_ssim_deterministic",
        "processed_images",
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

        writer.writerows(
            rows
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
            "Évaluer les VAE et CVAE "
            "sur Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
        help=(
            "Ensemble utilisé pour l'évaluation. "
            "Par défaut : validation."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Seed utilisée pour reconstruire "
            "le split train/validation. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Taille des batchs. "
            "Valeur par défaut : 128."
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
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed utilisée pour les opérations "
            "aléatoires de l'évaluation et "
            "de la génération. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--num-reconstruction-images",
        type=int,
        default=10,
        help=(
            "Nombre d'images utilisées "
            "dans les grilles de reconstruction."
        ),
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help=(
            "Nombre d'images générées "
            "pour chacune des dix classes."
        ),
    )

    # --max-test-batches reste accepté pour compatibilité
    # avec les anciennes commandes.
    parser.add_argument(
        "--max-eval-batches",
        "--max-test-batches",
        dest="max_eval_batches",
        type=int,
        default=None,
        help=(
            "Limite temporaire du nombre "
            "de batchs évalués. "
            "À utiliser uniquement pour "
            "un smoke test."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Chemin optionnel du CSV de sortie. "
            "Un chemin relatif est interprété depuis "
            "la racine du projet. Sans valeur, le fichier "
            "par défaut du split validation/test est utilisé."
        ),
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/vae_beta_01_seed42_final.pt",
            "checkpoints/vae_beta_1_seed42_final.pt",
            "checkpoints/vae_beta_4_seed42_final.pt",
            "checkpoints/cvae_beta_01_seed42_final.pt",
            "checkpoints/cvae_beta_1_seed42_final.pt",
            "checkpoints/cvae_beta_4_seed42_final.pt",
        ],
        help=(
            "Liste des checkpoints à évaluer."
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
    Vérifie les paramètres reçus.
    """

    if args.batch_size <= 0:

        raise ValueError(
            "batch-size doit être "
            "strictement positif."
        )

    if args.num_workers < 0:

        raise ValueError(
            "num-workers ne peut pas "
            "être négatif."
        )

    if args.num_reconstruction_images <= 0:

        raise ValueError(
            "num-reconstruction-images "
            "doit être positif."
        )

    if args.samples_per_class <= 0:

        raise ValueError(
            "samples-per-class doit "
            "être positif."
        )

    if (
        args.max_eval_batches is not None
        and args.max_eval_batches <= 0
    ):

        raise ValueError(
            "max-eval-batches doit être "
            "strictement positif."
        )


# ================================================================
# CHEMINS DES CHECKPOINTS
# ================================================================


def resolve_checkpoint_path(
    path_text: str,
) -> Path:
    """
    Convertit un chemin fourni dans le terminal en chemin absolu.
    """

    path = Path(
        path_text
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def resolve_output_path(
    path_text: str,
) -> Path:
    """
    Convertit un chemin de sortie en chemin absolu.

    Un chemin relatif est interprété depuis la racine du projet.
    """

    path = Path(
        path_text
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


# ================================================================
# PROGRAMME PRINCIPAL
# ================================================================


def main() -> None:
    """
    Point d'entrée principal de l'évaluation.
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

    evaluation_loader = create_evaluation_loader(
        split=args.split,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if args.output_csv is not None:

        metrics_path = resolve_output_path(
            args.output_csv
        )

    elif args.split == "validation":

        metrics_path = (
            VALIDATION_METRICS_PATH
        )

    else:

        metrics_path = (
            TEST_METRICS_PATH
        )

    print("=" * 78)

    print(
        "ÉVALUATION DES MODÈLES "
        "SUR FASHION-MNIST"
    )

    print("=" * 78)

    print(
        f"Split utilisé              : "
        f"{args.split}"
    )

    if args.split == "validation":

        print(
            f"Seed du split              : "
            f"{args.split_seed}"
        )

    print(
        f"Appareil utilisé           : "
        f"{device}"
    )

    print(
        f"Images disponibles         : "
        f"{len(evaluation_loader.dataset)}"
    )

    print(
        f"Taille des batchs          : "
        f"{args.batch_size}"
    )

    if args.max_eval_batches is not None:

        print(
            "Mode                       : "
            "TEST RAPIDE "
            f"({args.max_eval_batches} batchs)"
        )

    print("=" * 78)

    results: list[
        dict
    ] = []

    for checkpoint_text in args.checkpoints:

        checkpoint_path = (
            resolve_checkpoint_path(
                checkpoint_text
            )
        )

        print()

        print(
            f"Chargement : "
            f"{checkpoint_path.name}"
        )

        model, checkpoint = (
            load_model_from_checkpoint(
                checkpoint_path=(
                    checkpoint_path
                ),
                device=device,
            )
        )

        model_type = (
            checkpoint["model_type"]
        )

        beta = float(
            checkpoint["beta"]
        )

        # Même seed d'évaluation pour chaque checkpoint.
        # Cela rend l'estimation Monte-Carlo de l'ELBO
        # reproductible et utilise les mêmes tirages epsilon
        # dans le même ordre pour tous les modèles.
        set_random_seed(
            args.seed
        )

        metrics = evaluate_model(
            model=model,
            model_type=model_type,
            dataloader=evaluation_loader,
            device=device,
            beta=beta,
            max_batches=(
                args.max_eval_batches
            ),
        )

        reconstruction_path = (
            RECONSTRUCTION_DIR
            / (
                f"{checkpoint_path.stem}_"
                f"{args.split}_"
                "reconstructions.png"
            )
        )

        generation_path = (
            GENERATION_DIR
            / (
                f"{checkpoint_path.stem}_"
                f"generations_seed{args.seed}.png"
            )
        )

        save_reconstruction_examples(
            model=model,
            model_type=model_type,
            dataloader=evaluation_loader,
            device=device,
            output_path=(
                reconstruction_path
            ),
            num_images=(
                args.num_reconstruction_images
            ),
        )

        # Réinitialisation avant la génération qualitative :
        # chaque checkpoint reçoit le même seed de génération.
        set_random_seed(
            args.seed
        )

        save_generation_examples(
            model=model,
            model_type=model_type,
            device=device,
            output_path=(
                generation_path
            ),
            samples_per_class=(
                args.samples_per_class
            ),
        )

        configuration = checkpoint[
            "configuration"
        ]

        training_seed = configuration.get(
            "training_seed",
            configuration.get(
                "seed"
            ),
        )

        checkpoint_split_seed = (
            configuration.get(
                "split_seed"
            )
        )

        result_row = {
            "evaluation_split": (
                args.split
            ),

            "checkpoint": (
                checkpoint_path.name
            ),

            "model_type": (
                model_type
            ),

            "beta": (
                beta
            ),

            "training_seed": (
                training_seed
            ),

            "split_seed_protocol": (
                args.split_seed
                if args.split == "validation"
                else ""
            ),

            "checkpoint_split_seed": (
                checkpoint_split_seed
                if checkpoint_split_seed is not None
                else ""
            ),

            "evaluation_seed": (
                args.seed
            ),

            "best_epoch": (
                checkpoint["epoch"]
            ),

            "best_validation_loss": (
                checkpoint[
                    "best_validation_loss"
                ]
            ),

            "evaluation_total_sampled": (
                metrics["total"]
            ),

            "evaluation_reconstruction_bce_sampled": (
                metrics["reconstruction"]
            ),

            "evaluation_kl": (
                metrics["kl"]
            ),

            "evaluation_reconstruction_bce_deterministic": (
                metrics[
                    "deterministic_reconstruction_bce"
                ]
            ),

            "evaluation_mse_deterministic": (
                metrics["mse"]
            ),

            "evaluation_ssim_deterministic": (
                metrics["ssim"]
            ),

            "processed_images": (
                metrics[
                    "processed_samples"
                ]
            ),
        }

        results.append(
            result_row
        )

        print(
            f"Modèle                     : "
            f"{model_type}"
        )

        print(
            f"Beta                       : "
            f"{beta}"
        )

        print(
            f"Époque du checkpoint       : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Images évaluées            : "
            f"{metrics['processed_samples']}"
        )

        print(
            f"Loss totale (échantillonnée): "
            f"{metrics['total']:.4f}"
        )

        print(
            f"BCE (échantillonnée)       : "
            f"{metrics['reconstruction']:.4f}"
        )

        print(
            f"KL                         : "
            f"{metrics['kl']:.4f}"
        )

        print(
            f"BCE déterministe (z=mu)    : "
            f"{metrics['deterministic_reconstruction_bce']:.4f}"
        )

        print(
            f"MSE déterministe (z=mu)    : "
            f"{metrics['mse']:.6f}"
        )

        print(
            f"SSIM déterministe (z=mu)   : "
            f"{metrics['ssim']:.6f}"
        )

        print(
            "Reconstructions sauvegardées : "
            f"{reconstruction_path}"
        )

        print(
            "Générations sauvegardées     : "
            f"{generation_path}"
        )

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    save_metrics_csv(
        rows=results,
        output_path=metrics_path,
    )

    print()

    print("=" * 78)

    print(
        "ÉVALUATION TERMINÉE"
    )

    print(
        f"Split évalué               : "
        f"{args.split}"
    )

    print(
        f"Tableau des métriques      : "
        f"{metrics_path}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()