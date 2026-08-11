"""
Interpolation dans l'espace latent des modèles VAE et CVAE.

Objectif
--------
Ce script étudie la continuité de l'espace latent en interpolant entre
deux images Fashion-MNIST appartenant à une même classe.

Pour chaque classe sélectionnée :

1. deux images de test distinctes sont choisies de manière reproductible ;
2. chaque image est encodée vers le vecteur moyen latent ``mu`` ;
3. plusieurs points intermédiaires sont calculés par interpolation
   linéaire entre les deux vecteurs latents ;
4. chaque point latent est décodé en image ;
5. une grille PNG est sauvegardée.

Pourquoi utiliser ``mu`` ?
--------------------------
Le vecteur ``z`` produit par la reparamétrisation contient du bruit
aléatoire. Pour analyser proprement la continuité de l'espace latent,
nous utilisons le vecteur moyen ``mu``, qui est déterministe pour une
image donnée.

Pourquoi interpoler entre deux images de la même classe ?
---------------------------------------------------------
Cette stratégie permet de comparer équitablement le VAE et le CVAE.
Pour le CVAE, le même label est conservé pendant toute l'interpolation :
on observe donc comment le latent encode les variations de style à
l'intérieur d'une classe, sans mélanger artificiellement des labels
discrets.

Les mêmes paires d'images sont utilisées pour les six checkpoints afin
de rendre la comparaison entre bêta = 0.1, 1 et 4 aussi équitable que
possible.

Exécution depuis la racine du projet :

    python -m evaluation.interpolation

Test rapide recommandé :

    python -m evaluation.interpolation \
        --checkpoints checkpoints/cvae_beta_1.pt \
        --classes 0 5 \
        --steps 7 \
        --device cpu
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Final

import matplotlib

# Backend non interactif : les figures sont enregistrées sans ouvrir
# de fenêtre graphique.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from evaluation.evaluate import load_model_from_checkpoint


# ================================================================
# CHEMINS DU PROJET
# ================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "interpolations"
)


# ================================================================
# CLASSES FASHION-MNIST
# ================================================================

CLASS_NAMES: Final[list[str]] = [
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


# Les six modèles utilisés dans le projet.
DEFAULT_CHECKPOINTS: Final[list[str]] = [
    "checkpoints/vae_beta_01.pt",
    "checkpoints/vae_beta_1.pt",
    "checkpoints/vae_beta_4.pt",
    "checkpoints/cvae_beta_01.pt",
    "checkpoints/cvae_beta_1.pt",
    "checkpoints/cvae_beta_4.pt",
]


# Quatre classes suffisamment différentes pour obtenir une figure
# lisible sans la surcharger.
DEFAULT_CLASSES: Final[list[int]] = [
    0,  # T-shirt/top
    1,  # Trouser
    5,  # Sandal
    9,  # Ankle boot
]


def set_random_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires principales.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """
    Sélectionne l'appareil de calcul.

    Valeurs acceptées :
        auto
        cpu
        cuda
    """

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA a été demandé, mais aucun GPU CUDA "
                "n'est disponible."
            )

        return torch.device("cuda")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Appareil non reconnu : {device_name}"
    )


def resolve_project_path(path_text: str) -> Path:
    """
    Résout un chemin fourni depuis la racine du projet.
    """

    path = Path(path_text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def resolve_output_dir(path: Path) -> Path:
    """
    Transforme le dossier de sortie en chemin absolu.
    """

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def load_test_dataset() -> datasets.FashionMNIST:
    """
    Charge le jeu officiel de test Fashion-MNIST.
    """

    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    if len(dataset) != 10_000:
        raise RuntimeError(
            "Le jeu officiel de test devrait contenir "
            f"10 000 images, mais {len(dataset)} ont été trouvées."
        )

    return dataset


def validate_class_indices(
    classes: list[int],
) -> None:
    """
    Vérifie les indices de classes demandés.
    """

    if not classes:
        raise ValueError(
            "Au moins une classe doit être sélectionnée."
        )

    if len(classes) != len(set(classes)):
        raise ValueError(
            "La liste des classes contient des doublons."
        )

    invalid = [
        class_index
        for class_index in classes
        if class_index < 0 or class_index >= len(CLASS_NAMES)
    ]

    if invalid:
        raise ValueError(
            "Indices de classes invalides : "
            f"{invalid}. Les valeurs autorisées vont de 0 à 9."
        )


def select_image_pairs(
    dataset: datasets.FashionMNIST,
    classes: list[int],
    seed: int,
) -> dict[int, tuple[int, int]]:
    """
    Sélectionne deux images distinctes pour chaque classe.

    Les indices sont choisis une seule fois et réutilisés pour tous les
    modèles. Cela garantit une comparaison équitable.
    """

    targets = dataset.targets

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(
            targets,
            dtype=torch.long,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    pairs: dict[int, tuple[int, int]] = {}

    for class_index in classes:
        class_indices = torch.nonzero(
            targets == class_index,
            as_tuple=False,
        ).squeeze(1)

        if class_indices.numel() < 2:
            raise RuntimeError(
                f"La classe {class_index} ne contient pas "
                "au moins deux images."
            )

        permutation = torch.randperm(
            class_indices.numel(),
            generator=generator,
        )

        source_index = int(
            class_indices[permutation[0]].item()
        )

        target_index = int(
            class_indices[permutation[1]].item()
        )

        pairs[class_index] = (
            source_index,
            target_index,
        )

    return pairs


def get_pair_tensors(
    dataset: datasets.FashionMNIST,
    pair: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Charge les deux images correspondant à une paire d'indices.
    """

    source_image, _ = dataset[pair[0]]
    target_image, _ = dataset[pair[1]]

    return source_image, target_image


@torch.no_grad()
def encode_mean(
    model: nn.Module,
    model_type: str,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Encode les images et retourne le vecteur moyen latent ``mu``.
    """

    if model_type == "VAE":
        _, mu, _, _ = model(images)

    elif model_type == "CVAE":
        _, mu, _, _ = model(
            images,
            labels,
        )

    else:
        raise ValueError(
            f"Type de modèle non pris en charge : {model_type}."
        )

    return mu


def linear_interpolation(
    start: torch.Tensor,
    end: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule une interpolation linéaire entre deux vecteurs.

    La formule est :

        z(t) = (1 - t) * z_source + t * z_cible

    avec t compris entre 0 et 1.

    ``steps`` inclut les deux extrémités.
    """

    if steps < 2:
        raise ValueError(
            "steps doit être supérieur ou égal à 2."
        )

    if start.shape != end.shape:
        raise ValueError(
            "Les deux vecteurs latents doivent avoir "
            "la même forme."
        )

    alphas = torch.linspace(
        start=0.0,
        end=1.0,
        steps=steps,
        device=start.device,
        dtype=start.dtype,
    )

    alpha_column = alphas.unsqueeze(1)

    interpolated = (
        (1.0 - alpha_column) * start
        + alpha_column * end
    )

    return interpolated, alphas


@torch.no_grad()
def decode_latents(
    model: nn.Module,
    model_type: str,
    latents: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    """
    Décode les vecteurs interpolés.

    Pour le CVAE, le même label est conservé pendant toute
    l'interpolation.
    """

    if not hasattr(model, "decode"):
        raise AttributeError(
            "Le modèle ne possède pas de méthode 'decode'. "
            "L'interpolation nécessite une méthode decode."
        )

    if model_type == "VAE":
        images = model.decode(latents)

    elif model_type == "CVAE":
        labels = torch.full(
            size=(latents.shape[0],),
            fill_value=class_index,
            device=latents.device,
            dtype=torch.long,
        )

        images = model.decode(
            latents,
            labels,
        )

    else:
        raise ValueError(
            f"Type de modèle non pris en charge : {model_type}."
        )

    return images


@torch.no_grad()
def create_interpolation_row(
    model: nn.Module,
    model_type: str,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    class_index: int,
    steps: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Produit une interpolation complète pour une paire d'images.

    Returns
    -------
    source_image:
        Image originale de départ.

    decoded_images:
        Images décodées pour tous les coefficients d'interpolation.

    target_image:
        Image originale d'arrivée.

    alphas:
        Coefficients entre 0 et 1.
    """

    pair_images = torch.stack(
        [
            source_image,
            target_image,
        ],
        dim=0,
    ).to(device)

    pair_labels = torch.full(
        size=(2,),
        fill_value=class_index,
        device=device,
        dtype=torch.long,
    )

    mu = encode_mean(
        model=model,
        model_type=model_type,
        images=pair_images,
        labels=pair_labels,
    )

    source_mu = mu[0:1]
    target_mu = mu[1:2]

    interpolated_latents, alphas = linear_interpolation(
        start=source_mu,
        end=target_mu,
        steps=steps,
    )

    decoded_images = decode_latents(
        model=model,
        model_type=model_type,
        latents=interpolated_latents,
        class_index=class_index,
    )

    return (
        source_image.cpu(),
        decoded_images.cpu(),
        target_image.cpu(),
        alphas.cpu(),
    )


def beta_display(beta: float) -> str:
    """
    Formate bêta pour les titres.
    """

    if float(beta).is_integer():
        return str(int(beta))

    return str(beta).replace(".", ",")


def tensor_to_image(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convertit un tenseur [1, 28, 28] en tableau 2D NumPy.
    """

    array = (
        tensor.detach()
        .cpu()
        .squeeze(0)
        .numpy()
    )

    return array


def save_interpolation_figure(
    rows: list[
        tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    model_type: str,
    beta: float,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Sauvegarde une grille d'interpolation.

    Chaque ligne contient :

        Original A | t=0 ... t=1 | Original B
    """

    if not rows:
        raise ValueError(
            "Aucune interpolation n'est disponible."
        )

    number_of_rows = len(rows)
    steps = int(rows[0][2].shape[0])
    number_of_columns = steps + 2

    figure_width = max(
        12.0,
        number_of_columns * 1.45,
    )

    figure_height = max(
        3.0,
        number_of_rows * 2.0,
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    for row_index, (
        class_index,
        source_image,
        decoded_images,
        target_image,
        alphas,
    ) in enumerate(rows):
        # Image originale de départ.
        axes[row_index, 0].imshow(
            tensor_to_image(source_image),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        # Images décodées au fil de l'interpolation.
        for step_index in range(steps):
            axes[
                row_index,
                step_index + 1,
            ].imshow(
                tensor_to_image(
                    decoded_images[step_index]
                ),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

        # Image originale d'arrivée.
        axes[
            row_index,
            number_of_columns - 1,
        ].imshow(
            tensor_to_image(target_image),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        for column_index in range(number_of_columns):
            axes[
                row_index,
                column_index,
            ].set_xticks([])

            axes[
                row_index,
                column_index,
            ].set_yticks([])

        axes[row_index, 0].set_ylabel(
            f"{class_index} — {CLASS_NAMES[class_index]}",
            fontsize=9,
        )

        # Titres des colonnes sur la première ligne uniquement.
        if row_index == 0:
            axes[row_index, 0].set_title(
                "Original A",
                fontsize=9,
            )

            for step_index, alpha in enumerate(
                alphas.tolist()
            ):
                axes[
                    row_index,
                    step_index + 1,
                ].set_title(
                    f"t={alpha:.2f}",
                    fontsize=8,
                )

            axes[
                row_index,
                number_of_columns - 1,
            ].set_title(
                "Original B",
                fontsize=9,
            )

    figure.suptitle(
        (
            f"{model_type} — interpolation latente linéaire "
            f"— β = {beta_display(beta)}"
        ),
        fontsize=13,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_metadata_csv(
    output_path: Path,
    checkpoint_name: str,
    model_type: str,
    beta: float,
    pairs: dict[int, tuple[int, int]],
    rows: list[
        tuple[
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ],
) -> None:
    """
    Sauvegarde les informations nécessaires pour reproduire la figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "checkpoint",
        "model_type",
        "beta",
        "class_index",
        "class_name",
        "source_test_index",
        "target_test_index",
        "steps",
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

        for (
            class_index,
            _,
            decoded_images,
            _,
            _,
        ) in rows:
            source_index, target_index = pairs[
                class_index
            ]

            writer.writerow(
                {
                    "checkpoint": checkpoint_name,
                    "model_type": model_type,
                    "beta": beta,
                    "class_index": class_index,
                    "class_name": CLASS_NAMES[
                        class_index
                    ],
                    "source_test_index": source_index,
                    "target_test_index": target_index,
                    "steps": int(
                        decoded_images.shape[0]
                    ),
                }
            )


def process_checkpoint(
    checkpoint_path: Path,
    dataset: datasets.FashionMNIST,
    pairs: dict[int, tuple[int, int]],
    classes: list[int],
    steps: int,
    device: torch.device,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    Produit la figure et le CSV pour un checkpoint.
    """

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model_type = str(
        checkpoint["model_type"]
    )

    beta = float(
        checkpoint["beta"]
    )

    model.eval()

    rows = []

    for class_index in classes:
        source_image, target_image = get_pair_tensors(
            dataset=dataset,
            pair=pairs[class_index],
        )

        (
            source_image,
            decoded_images,
            target_image,
            alphas,
        ) = create_interpolation_row(
            model=model,
            model_type=model_type,
            source_image=source_image,
            target_image=target_image,
            class_index=class_index,
            steps=steps,
            device=device,
        )

        rows.append(
            (
                class_index,
                source_image,
                decoded_images,
                target_image,
                alphas,
            )
        )

    figure_path = (
        output_dir
        / f"{checkpoint_path.stem}_interpolation.png"
    )

    metadata_path = (
        output_dir
        / f"{checkpoint_path.stem}_interpolation.csv"
    )

    save_interpolation_figure(
        rows=rows,
        model_type=model_type,
        beta=beta,
        output_path=figure_path,
        dpi=dpi,
    )

    save_metadata_csv(
        output_path=metadata_path,
        checkpoint_name=checkpoint_path.name,
        model_type=model_type,
        beta=beta,
        pairs=pairs,
        rows=rows,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [
        figure_path,
        metadata_path,
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Définit les arguments disponibles depuis PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Créer des interpolations dans les espaces latents "
            "des VAE et CVAE."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help=(
            "Liste des checkpoints à analyser. "
            "Par défaut, les six modèles sont utilisés."
        ),
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=DEFAULT_CLASSES,
        help=(
            "Classes utilisées pour les interpolations. "
            "Par défaut : 0 1 5 9."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help=(
            "Nombre de points latents entre t=0 et t=1, "
            "extrémités incluses. Valeur par défaut : 9."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Graine utilisée pour choisir les paires d'images. "
            "Valeur par défaut : 42."
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Appareil de calcul. Valeur par défaut : auto."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help=(
            "Résolution des figures. Valeur par défaut : 200."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Dossier de sortie. Valeur par défaut : "
            "results/interpolations."
        ),
    )

    return parser


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """
    Vérifie les arguments reçus.
    """

    validate_class_indices(
        args.classes
    )

    if args.steps < 2:
        raise ValueError(
            "steps doit être supérieur ou égal à 2."
        )

    if args.dpi <= 0:
        raise ValueError(
            "dpi doit être strictement positif."
        )


def main() -> None:
    """
    Point d'entrée principal du script.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_arguments(args)
    set_random_seed(args.seed)

    device = select_device(
        args.device
    )

    output_dir = resolve_output_dir(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_test_dataset()

    # Les paires sont créées une seule fois, avant de charger les
    # modèles. Les six checkpoints utilisent donc exactement les
    # mêmes images de départ et d'arrivée.
    pairs = select_image_pairs(
        dataset=dataset,
        classes=args.classes,
        seed=args.seed,
    )

    checkpoint_paths = [
        resolve_project_path(path_text)
        for path_text in args.checkpoints
    ]

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}"
            )

    print("=" * 74)
    print("INTERPOLATION DANS L'ESPACE LATENT — VAE ET CVAE")
    print("=" * 74)
    print(f"Appareil utilisé       : {device}")
    print(f"Nombre de checkpoints : {len(checkpoint_paths)}")
    print(f"Classes sélectionnées : {args.classes}")
    print(f"Étapes par interpolation : {args.steps}")
    print(f"Graine                 : {args.seed}")
    print(f"Dossier de sortie      : {output_dir}")
    print("=" * 74)

    print()
    print("Paires d'images du jeu de test :")

    for class_index in args.classes:
        source_index, target_index = pairs[
            class_index
        ]

        print(
            f"  Classe {class_index} "
            f"({CLASS_NAMES[class_index]}) : "
            f"{source_index} -> {target_index}"
        )

    created_files: list[Path] = []

    for index, checkpoint_path in enumerate(
        checkpoint_paths,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(checkpoint_paths)}] "
            f"{checkpoint_path.name}"
        )

        checkpoint_files = process_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            pairs=pairs,
            classes=args.classes,
            steps=args.steps,
            device=device,
            output_dir=output_dir,
            dpi=args.dpi,
        )

        created_files.extend(
            checkpoint_files
        )

        print(
            f"  -> {len(checkpoint_files)} fichier(s) créé(s)."
        )

    print()
    print("=" * 74)
    print("INTERPOLATION TERMINÉE")
    print(f"Nombre de fichiers créés : {len(created_files)}")
    print("=" * 74)

    for file_path in created_files:
        print(
            f"{file_path} -> "
            f"{file_path.stat().st_size} octets"
        )


if __name__ == "__main__":
    main()