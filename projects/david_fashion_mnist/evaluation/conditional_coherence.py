"""
Évaluation de la cohérence conditionnelle des CVAE Fashion-MNIST.

Objectif
--------
Mesurer quantitativement si les images générées par un CVAE
correspondent réellement à la classe demandée.

Le protocole utilise un classifieur Fashion-MNIST indépendant,
préalablement entraîné sur les vraies images du dataset.

Pour chaque CVAE :

1. on demande une classe Fashion-MNIST ;
2. on génère des images avec cette condition ;
3. le classifieur indépendant prédit la classe des images générées ;
4. on compare la classe prédite avec la classe demandée.

La métrique principale est :

    conditional_accuracy
    =
    nombre d'images reconnues comme la classe demandée
    --------------------------------------------------
               nombre total d'images générées

Comparaison contrôlée
---------------------
Une seule banque de vecteurs latents z ~ N(0, I) est créée.

Les mêmes vecteurs z sont ensuite réutilisés :

- pour les dix classes ;
- pour CVAE beta = 0.1 ;
- pour CVAE beta = 1 ;
- pour CVAE beta = 4.

Ainsi, les différences observées entre les modèles ne proviennent
pas de tirages aléatoires différents du vecteur latent.

Important
---------
Le jeu officiel de test Fashion-MNIST n'est pas utilisé.

Cette évaluation utilise uniquement :

- les checkpoints des CVAE ;
- le checkpoint du classifieur indépendant ;
- des vecteurs latents générés artificiellement.

Exemple
-------
    python -m evaluation.conditional_coherence --device cuda

Smoke test
----------
    python -m evaluation.conditional_coherence \
        --device cuda \
        --samples-per-class 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
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
    / "conditional_coherence"
)

DEFAULT_CLASSIFIER_CHECKPOINT = (
    CHECKPOINT_DIR
    / "fashion_classifier_seed42_final.pt"
)

DEFAULT_CVAE_CHECKPOINTS = [
    CHECKPOINT_DIR
    / "cvae_beta_01_seed42_final.pt",

    CHECKPOINT_DIR
    / "cvae_beta_1_seed42_final.pt",

    CHECKPOINT_DIR
    / "cvae_beta_4_seed42_final.pt",
]

DEFAULT_MLFLOW_EXPERIMENT = (
    "fashion_mnist_conditional_coherence"
)

DEFAULT_MLFLOW_RUN_NAME = (
    "cvae_conditional_coherence_seed42"
)


# ================================================================
# OUTILS GÉNÉRAUX
# ================================================================


def resolve_project_path(
    path: Path,
) -> Path:
    """
    Transforme un chemin relatif en chemin absolu du sous-projet.

    Un chemin déjà absolu est conservé tel quel.
    """

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def beta_to_tag(
    beta: float,
) -> str:
    """
    Transforme beta en texte utilisable dans des noms de métriques.

    Exemples
    --------
    0.1 -> "01"
    1.0 -> "1"
    4.0 -> "4"
    """

    return (
        f"{beta:g}"
        .replace(".", "")
    )


# ================================================================
# CHARGEMENT D'UN CVAE
# ================================================================


def load_cvae(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[CVAE, dict]:
    """
    Charge un checkpoint CVAE.

    Parameters
    ----------
    checkpoint_path:
        Checkpoint du modèle.

    device:
        CPU ou CUDA.

    Returns
    -------
    model:
        CVAE chargé en mode évaluation.

    checkpoint:
        Dictionnaire complet du checkpoint.
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

    latent_dim = configuration.get(
        "latent_dim",
        16,
    )

    hidden_dim = configuration.get(
        "hidden_dim",
        256,
    )

    num_classes = configuration.get(
        "num_classes",
        10,
    )

    if num_classes != len(FASHION_MNIST_CLASSES):
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
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ================================================================
# BANQUE LATENTE CONTRÔLÉE
# ================================================================


def create_shared_latent_vectors(
    samples_per_class: int,
    latent_dim: int,
    seed: int,
) -> Tensor:
    """
    Crée les vecteurs latents utilisés par tous les modèles.

    Une seule matrice est générée :

        [samples_per_class, latent_dim]

    Cette même matrice est ensuite réutilisée :

    - pour chaque classe ;
    - pour chaque valeur de beta.

    Cela permet une comparaison contrôlée.
    """

    generator = (
        torch.Generator(
            device="cpu"
        )
        .manual_seed(seed)
    )

    latent_vectors = torch.randn(
        samples_per_class,
        latent_dim,
        generator=generator,
        device="cpu",
    )

    return latent_vectors


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

    Convention de la matrice
    ------------------------
    Lignes :
        classe demandée au CVAE.

    Colonnes :
        classe prédite par le classifieur indépendant.

    Exemple
    -------
    matrix[9, 7] indique combien d'images demandées comme
    "Ankle boot" ont été reconnues comme "Sneaker".
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

    # Chaque classe reçoit exactement la même banque de z.
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

            # Génération contrôlée :
            #
            # on appelle directement decode() afin de fournir
            # nous-mêmes les vecteurs z.
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

    if confusion_matrix.shape != expected_shape:
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
        dict[str, float | int | str]
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
) -> None:
    """
    Sauvegarde une grille de générations contrôlées.

    Les colonnes correspondent aux mêmes z pour toutes les classes.

    Les mêmes premiers z sont aussi utilisés pour tous les CVAE.

    Cela donne une comparaison qualitative beaucoup plus rigoureuse
    que des tirages indépendants pour chaque modèle.
    """

    if samples_per_class <= 0:
        return

    available_samples = (
        shared_latent_vectors.shape[0]
    )

    samples_per_class = min(
        samples_per_class,
        available_samples,
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
            generated.detach().cpu()
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
            max(8, samples_per_class),
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
        "Générations CVAE contrôlées\n"
        "Même vecteur latent z dans chaque colonne"
    )

    axis.axis(
        "off"
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
# FIGURE DE MATRICE
# ================================================================


def save_conditional_confusion_figure(
    normalized_confusion_matrix: np.ndarray,
    beta: float,
    output_path: Path,
) -> None:
    """
    Sauvegarde la matrice :

        classe demandée -> classe prédite

    sous forme graphique.
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
        f"beta = {beta:g}"
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
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ================================================================
# CSV RÉCAPITULATIF GLOBAL
# ================================================================


def save_summary_csv(
    rows: list[
        dict[str, float | int | str]
    ],
    output_path: Path,
) -> None:
    """
    Sauvegarde une ligne récapitulative par CVAE.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "checkpoint",
        "model_type",
        "beta",
        "best_epoch",
        "samples_per_class",
        "total_generated_images",
        "correct_predictions",
        "conditional_accuracy",
        "latent_seed",
        "same_latents_across_models",
        "same_latents_across_classes",
        "classifier_checkpoint",
        "classifier_validation_accuracy",
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
# CSV PAR CLASSE
# ================================================================


def save_per_class_csv(
    rows: list[
        dict[str, float | int | str]
    ],
    output_path: Path,
) -> None:
    """
    Sauvegarde les résultats conditionnels classe par classe.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "checkpoint",
        "beta",
        "class_index",
        "class_name",
        "requested_samples",
        "correct_predictions",
        "conditional_accuracy",
        "latent_seed",
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
# AFFICHAGE D'UN MODÈLE
# ================================================================


def print_model_results(
    checkpoint_name: str,
    beta: float,
    epoch: int,
    global_metrics: dict[
        str,
        float | int
    ],
    per_class_metrics: list[
        dict[str, float | int | str]
    ],
) -> None:
    """
    Affiche les résultats d'un CVAE.
    """

    print("=" * 88)

    print(
        f"CVAE : {checkpoint_name}"
    )

    print("=" * 88)

    print(
        f"Beta                       : "
        f"{beta:g}"
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

    print("-" * 88)

    print(
        f"{'ID':>2}  "
        f"{'Classe demandée':<14} "
        f"{'Images':>8} "
        f"{'Correct':>8} "
        f"{'Cohérence':>11}"
    )

    print("-" * 88)

    for row in per_class_metrics:

        print(
            f"{int(row['class_index']):>2}  "
            f"{str(row['class_name']):<14} "
            f"{int(row['requested_samples']):>8} "
            f"{int(row['correct_predictions']):>8} "
            f"{float(row['conditional_accuracy']):>10.2%}"
        )

    print("=" * 88)


# ================================================================
# ARGUMENTS
# ================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments disponibles.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Mesurer la cohérence conditionnelle "
            "des CVAE Fashion-MNIST."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=DEFAULT_CVAE_CHECKPOINTS,
        help=(
            "Checkpoints CVAE à comparer. "
            "Par défaut : beta 0.1, 1 et 4."
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
        "--samples-per-class",
        type=int,
        default=1000,
        help=(
            "Nombre d'images générées pour chaque classe. "
            "Valeur par défaut : 1000."
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
            "qualitatives contrôlées. "
            "Utiliser 0 pour désactiver les grilles."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed utilisé pour créer la banque de z. "
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
            "Nom du run MLflow de comparaison."
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

    classifier_checkpoint_path = (
        resolve_project_path(
            args.classifier_checkpoint
        )
    )

    classifier, classifier_checkpoint = (
        load_classifier(
            checkpoint_path=(
                classifier_checkpoint_path
            ),
            device=device,
        )
    )

    classifier_validation_accuracy = (
        classifier_checkpoint.get(
            "validation_accuracy"
        )
    )

    # ------------------------------------------------------------
    # Chargement de tous les CVAE avant la génération.
    #
    # Cela permet de vérifier qu'ils utilisent la même dimension
    # latente avant de construire la banque commune de z.
    # ------------------------------------------------------------

    loaded_models: list[
        tuple[
            Path,
            CVAE,
            dict,
        ]
    ] = []

    latent_dimensions: set[int] = set()

    for checkpoint_argument in args.checkpoints:

        checkpoint_path = resolve_project_path(
            checkpoint_argument
        )

        cvae, checkpoint = load_cvae(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        latent_dimensions.add(
            cvae.latent_dim
        )

        loaded_models.append(
            (
                checkpoint_path,
                cvae,
                checkpoint,
            )
        )

    if len(latent_dimensions) != 1:
        raise ValueError(
            "Tous les CVAE doivent avoir la même dimension latente "
            "pour une comparaison contrôlée. "
            f"Dimensions observées : {sorted(latent_dimensions)}"
        )

    latent_dim = next(
        iter(latent_dimensions)
    )

    shared_latent_vectors = (
        create_shared_latent_vectors(
            samples_per_class=(
                args.samples_per_class
            ),
            latent_dim=latent_dim,
            seed=args.seed,
        )
    )

    total_images_per_model = (
        args.samples_per_class
        * len(FASHION_MNIST_CLASSES)
    )

    print("=" * 88)

    print(
        "ÉVALUATION DE LA COHÉRENCE "
        "CONDITIONNELLE DES CVAE"
    )

    print("=" * 88)

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
        f"{classifier_validation_accuracy:.4%}"
    )

    print(
        f"Nombre de CVAE                    : "
        f"{len(loaded_models)}"
    )

    print(
        f"Dimension latente                 : "
        f"{latent_dim}"
    )

    print(
        f"Images par classe                 : "
        f"{args.samples_per_class}"
    )

    print(
        f"Images générées par CVAE          : "
        f"{total_images_per_model}"
    )

    print(
        f"Seed des vecteurs latents         : "
        f"{args.seed}"
    )

    print(
        "Mêmes z entre les CVAE            : "
        "OUI"
    )

    print(
        "Mêmes z entre les classes         : "
        "OUI"
    )

    print(
        "Jeu officiel de test utilisé      : "
        "NON"
    )

    print("=" * 88)

    summary_rows: list[
        dict[str, float | int | str]
    ] = []

    all_per_class_rows: list[
        dict[str, float | int | str]
    ] = []

    # ------------------------------------------------------------
    # Configuration MLflow
    # ------------------------------------------------------------

    tracking_uri: Optional[str] = None

    if not args.disable_mlflow:

        tracking_uri = configure_mlflow(
            tracking_uri=(
                args.mlflow_tracking_uri
            ),
            experiment_name=(
                args.mlflow_experiment_name
            ),
        )

    # ------------------------------------------------------------
    # Fonction interne exécutant réellement les trois évaluations.
    # ------------------------------------------------------------

    def run_evaluations() -> None:
        """
        Évalue successivement tous les CVAE chargés.
        """

        for (
            checkpoint_path,
            cvae,
            checkpoint,
        ) in loaded_models:

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

            epoch = int(
                checkpoint.get(
                    "epoch",
                    0,
                )
            )

            beta_tag = beta_to_tag(
                beta
            )

            print(
                f"\nÉvaluation de "
                f"{checkpoint_path.name}..."
            )

            confusion_matrix = (
                evaluate_conditional_coherence(
                    cvae=cvae,
                    classifier=classifier,
                    shared_latent_vectors=(
                        shared_latent_vectors
                    ),
                    device=device,
                    generation_batch_size=(
                        args.generation_batch_size
                    ),
                )
            )

            (
                global_metrics,
                per_class_metrics,
            ) = calculate_conditional_metrics(
                confusion_matrix
            )

            normalized_confusion = (
                normalize_confusion_matrix(
                    confusion_matrix
                )
            )

            # ----------------------------------------------------
            # Fichiers spécifiques au modèle
            # ----------------------------------------------------

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
                confusion_matrix=(
                    confusion_matrix
                ),
                output_path=(
                    raw_matrix_path
                ),
            )

            save_confusion_matrix_csv(
                confusion_matrix=(
                    normalized_confusion
                ),
                output_path=(
                    normalized_matrix_path
                ),
                decimal_places=6,
            )

            save_conditional_confusion_figure(
                normalized_confusion_matrix=(
                    normalized_confusion
                ),
                beta=beta,
                output_path=(
                    matrix_figure_path
                ),
            )

            if args.grid_samples_per_class > 0:

                save_controlled_generation_grid(
                    cvae=cvae,
                    shared_latent_vectors=(
                        shared_latent_vectors
                    ),
                    samples_per_class=(
                        args.grid_samples_per_class
                    ),
                    device=device,
                    output_path=(
                        generation_grid_path
                    ),
                )

            # ----------------------------------------------------
            # Résumé global
            # ----------------------------------------------------

            summary_rows.append(
                {
                    "checkpoint": (
                        checkpoint_path.name
                    ),
                    "model_type": "CVAE",
                    "beta": beta,
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
                    "same_latents_across_models": (
                        True
                    ),
                    "same_latents_across_classes": (
                        True
                    ),
                    "classifier_checkpoint": (
                        classifier_checkpoint_path.name
                    ),
                    "classifier_validation_accuracy": (
                        classifier_validation_accuracy
                    ),
                }
            )

            # ----------------------------------------------------
            # Résultats par classe
            # ----------------------------------------------------

            for class_row in per_class_metrics:

                all_per_class_rows.append(
                    {
                        "checkpoint": (
                            checkpoint_path.name
                        ),
                        "beta": beta,
                        **class_row,
                        "latent_seed": (
                            args.seed
                        ),
                    }
                )

            print_model_results(
                checkpoint_name=(
                    checkpoint_path.name
                ),
                beta=beta,
                epoch=epoch,
                global_metrics=(
                    global_metrics
                ),
                per_class_metrics=(
                    per_class_metrics
                ),
            )

            # ----------------------------------------------------
            # MLflow
            # ----------------------------------------------------

            if not args.disable_mlflow:

                metric_prefix = (
                    f"beta_{beta_tag}"
                )

                mlflow.log_metric(
                    (
                        "conditional_accuracy_"
                        f"{metric_prefix}"
                    ),
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
                            f"{metric_prefix}_"
                            f"class_{class_index}"
                        ),
                        float(
                            class_row[
                                "conditional_accuracy"
                            ]
                        ),
                    )

                mlflow.log_artifact(
                    str(
                        raw_matrix_path
                    ),
                    artifact_path=(
                        checkpoint_path.stem
                    ),
                )

                mlflow.log_artifact(
                    str(
                        normalized_matrix_path
                    ),
                    artifact_path=(
                        checkpoint_path.stem
                    ),
                )

                mlflow.log_artifact(
                    str(
                        matrix_figure_path
                    ),
                    artifact_path=(
                        checkpoint_path.stem
                    ),
                )

                if (
                    args.grid_samples_per_class > 0
                    and generation_grid_path.exists()
                ):

                    mlflow.log_artifact(
                        str(
                            generation_grid_path
                        ),
                        artifact_path=(
                            checkpoint_path.stem
                        ),
                    )

    # ------------------------------------------------------------
    # Exécution avec ou sans MLflow
    # ------------------------------------------------------------

    if args.disable_mlflow:

        run_evaluations()

    else:

        with mlflow.start_run(
            run_name=(
                args.mlflow_run_name
            )
        ) as active_run:

            print(
                f"MLflow run ID                     : "
                f"{active_run.info.run_id}"
            )

            print(
                f"MLflow tracking URI               : "
                f"{tracking_uri}"
            )

            print("=" * 88)

            mlflow.log_params(
                {
                    "evaluation": (
                        "conditional_coherence"
                    ),
                    "dataset": (
                        "Fashion-MNIST"
                    ),
                    "model_type": (
                        "CVAE"
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
                    "same_latents_across_models": (
                        True
                    ),
                    "same_latents_across_classes": (
                        True
                    ),
                    "classifier_checkpoint": (
                        classifier_checkpoint_path.name
                    ),
                    "classifier_validation_accuracy": (
                        classifier_validation_accuracy
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
                        "conditional_coherence"
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

            # ----------------------------------------------------
            # CSV finaux
            # ----------------------------------------------------

            summary_path = (
                output_dir
                / "conditional_coherence_summary.csv"
            )

            per_class_path = (
                output_dir
                / "conditional_coherence_per_class.csv"
            )

            save_summary_csv(
                rows=summary_rows,
                output_path=summary_path,
            )

            save_per_class_csv(
                rows=all_per_class_rows,
                output_path=per_class_path,
            )

            mlflow.log_artifact(
                str(summary_path),
                artifact_path=(
                    "comparison"
                ),
            )

            mlflow.log_artifact(
                str(per_class_path),
                artifact_path=(
                    "comparison"
                ),
            )

            # Meilleur score observé parmi les CVAE.
            if summary_rows:

                best_accuracy = max(
                    float(
                        row[
                            "conditional_accuracy"
                        ]
                    )
                    for row in summary_rows
                )

                mlflow.log_metric(
                    "best_conditional_accuracy",
                    best_accuracy,
                )

    # ------------------------------------------------------------
    # Si MLflow est désactivé, il faut aussi créer les CSV finaux.
    # ------------------------------------------------------------

    if args.disable_mlflow:

        summary_path = (
            output_dir
            / "conditional_coherence_summary.csv"
        )

        per_class_path = (
            output_dir
            / "conditional_coherence_per_class.csv"
        )

        save_summary_csv(
            rows=summary_rows,
            output_path=summary_path,
        )

        save_per_class_csv(
            rows=all_per_class_rows,
            output_path=per_class_path,
        )

    # ------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------

    print("\n" + "=" * 88)

    print(
        "COMPARAISON FINALE DE LA "
        "COHÉRENCE CONDITIONNELLE"
    )

    print("=" * 88)

    print(
        f"{'Beta':>6} "
        f"{'Images':>10} "
        f"{'Correctes':>12} "
        f"{'Conditional accuracy':>22}"
    )

    print("-" * 88)

    for row in summary_rows:

        print(
            f"{float(row['beta']):>6g} "
            f"{int(row['total_generated_images']):>10} "
            f"{int(row['correct_predictions']):>12} "
            f"{float(row['conditional_accuracy']):>21.2%}"
        )

    print("=" * 88)

    print(
        f"Résumé CSV                : "
        f"{output_dir / 'conditional_coherence_summary.csv'}"
    )

    print(
        f"Résultats par classe      : "
        f"{output_dir / 'conditional_coherence_per_class.csv'}"
    )

    print(
        "Jeu officiel de test     : "
        "NON UTILISÉ"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()