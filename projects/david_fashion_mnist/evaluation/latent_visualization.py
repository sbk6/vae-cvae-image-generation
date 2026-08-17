"""
Visualisation 2D de l'espace latent des six modèles VAE/CVAE.

Le script :
- charge les six checkpoints ;
- utilise un sous-ensemble équilibré du jeu de test Fashion-MNIST ;
- extrait le vecteur moyen latent mu ;
- crée une projection PCA et une projection UMAP pour chaque modèle ;
- sauvegarde les graphiques et les coordonnées dans results/latent_spaces.

Exécution depuis la racine du projet :
    python -m evaluation.latent_visualization
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from evaluation.evaluate import load_model_from_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "latent_spaces"

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

DEFAULT_CHECKPOINTS = [
    "checkpoints/vae_beta_01.pt",
    "checkpoints/vae_beta_1.pt",
    "checkpoints/vae_beta_4.pt",
    "checkpoints/cvae_beta_01.pt",
    "checkpoints/cvae_beta_1.pt",
    "checkpoints/cvae_beta_4.pt",
]


def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires principales."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    """Choisit CPU ou CUDA."""
    if name == "cpu":
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA a été demandé mais n'est pas disponible."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def resolve_project_path(path_text: str) -> Path:
    """Résout un chemin relatif depuis la racine du projet."""
    path = Path(path_text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def create_test_dataset() -> datasets.FashionMNIST:
    """Charge les 10 000 images officielles de test."""
    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    if len(dataset) != 10_000:
        raise RuntimeError(
            f"10 000 images de test étaient attendues, "
            f"mais {len(dataset)} ont été trouvées."
        )

    return dataset


def balanced_indices(
    targets: torch.Tensor,
    max_samples: int,
    seed: int,
) -> list[int]:
    """
    Sélectionne un sous-ensemble déterministe et équilibré.

    Avec 3 000 images, on prend 300 images par classe.
    """
    if max_samples <= 0 or max_samples > len(targets):
        raise ValueError(
            "max_samples doit être compris entre 1 et "
            f"{len(targets)}."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    num_classes = len(CLASS_NAMES)
    base = max_samples // num_classes
    remainder = max_samples % num_classes

    selected: list[int] = []

    for class_index in range(num_classes):
        class_indices = torch.nonzero(
            targets == class_index,
            as_tuple=False,
        ).squeeze(1)

        requested = base + (1 if class_index < remainder else 0)

        permutation = torch.randperm(
            class_indices.numel(),
            generator=generator,
        )

        chosen = class_indices[permutation[:requested]]
        selected.extend(chosen.tolist())

    # Mélange final reproductible.
    order = torch.randperm(
        len(selected),
        generator=generator,
    ).tolist()

    return [selected[index] for index in order]


def create_latent_loader(
    dataset: datasets.FashionMNIST,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    """Crée le DataLoader utilisé par tous les modèles."""
    targets = dataset.targets

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long)

    indices = balanced_indices(
        targets=targets,
        max_samples=max_samples,
        seed=seed,
    )

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def extract_mu(
    model,
    model_type: str,
    dataloader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait mu, et non z, afin d'éviter le bruit aléatoire de la
    reparamétrisation pendant la visualisation.
    """
    model.eval()

    mu_batches = []
    label_batches = []

    for images, labels in tqdm(
        dataloader,
        desc=description,
        leave=False,
    ):
        images = images.to(device)
        labels_device = labels.to(device)

        if model_type == "VAE":
            _, mu, _, _ = model(images)

        elif model_type == "CVAE":
            _, mu, _, _ = model(images, labels_device)

        else:
            raise ValueError(
                f"Type de modèle inconnu : {model_type}"
            )

        mu_batches.append(mu.cpu())
        label_batches.append(labels.cpu())

    latent = torch.cat(mu_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()

    return latent, labels


def project_pca(
    latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Projette l'espace latent en deux dimensions avec PCA."""
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(latent)
    return coordinates, pca.explained_variance_ratio_


def project_umap(
    latent: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Projette l'espace latent en deux dimensions avec UMAP."""
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "umap-learn n'est pas installé. "
            "Commande : pip install umap-learn"
        ) from error

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
    )

    return reducer.fit_transform(latent)


def save_coordinates(
    coordinates: np.ndarray,
    labels: np.ndarray,
    method: str,
    output_path: Path,
) -> None:
    """Sauvegarde les coordonnées projetées dans un CSV."""
    dataframe = pd.DataFrame(
        {
            "component_1": coordinates[:, 0],
            "component_2": coordinates[:, 1],
            "label": labels.astype(int),
            "class_name": [
                CLASS_NAMES[int(label)]
                for label in labels
            ],
            "method": method,
        }
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def save_plot(
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Crée un graphique 2D avec une catégorie par classe."""
    figure, axis = plt.subplots(figsize=(9, 7))

    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_index

        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=12,
            alpha=0.65,
            label=f"{class_index} — {class_name}",
        )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def beta_text(beta: float) -> str:
    """Formate bêta pour les titres."""
    if float(beta).is_integer():
        return str(int(beta))

    return str(beta).replace(".", ",")


def process_checkpoint(
    checkpoint_path: Path,
    dataloader: DataLoader,
    device: torch.device,
    methods: list[str],
    output_dir: Path,
    seed: int,
    dpi: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> list[Path]:
    """Crée toutes les projections demandées pour un checkpoint."""
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model_type = str(checkpoint["model_type"])
    beta = float(checkpoint["beta"])

    latent, labels = extract_mu(
        model=model,
        model_type=model_type,
        dataloader=dataloader,
        device=device,
        description=f"Latents {checkpoint_path.stem}",
    )

    expected_dim = int(
        checkpoint["configuration"]["latent_dim"]
    )

    if latent.shape[1] != expected_dim:
        raise RuntimeError(
            f"Dimension latente attendue : {expected_dim}; "
            f"obtenue : {latent.shape[1]}."
        )

    title_base = (
        f"{model_type} — espace latent — β = {beta_text(beta)}"
    )

    created: list[Path] = []

    if "pca" in methods:
        coordinates, variance = project_pca(latent)

        png_path = (
            output_dir
            / f"{checkpoint_path.stem}_latent_pca.png"
        )
        csv_path = (
            output_dir
            / f"{checkpoint_path.stem}_latent_pca.csv"
        )
        info_path = (
            output_dir
            / f"{checkpoint_path.stem}_latent_pca_info.txt"
        )

        save_plot(
            coordinates=coordinates,
            labels=labels,
            title=f"{title_base} — PCA",
            x_label="Composante principale 1",
            y_label="Composante principale 2",
            output_path=png_path,
            dpi=dpi,
        )

        save_coordinates(
            coordinates=coordinates,
            labels=labels,
            method="PCA",
            output_path=csv_path,
        )

        info_path.write_text(
            "\n".join(
                [
                    f"checkpoint={checkpoint_path.name}",
                    f"model_type={model_type}",
                    f"beta={beta}",
                    f"samples={latent.shape[0]}",
                    f"latent_dim={latent.shape[1]}",
                    f"pca_variance_1={variance[0]}",
                    f"pca_variance_2={variance[1]}",
                    f"pca_variance_total={variance.sum()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        created.extend(
            [png_path, csv_path, info_path]
        )

    if "umap" in methods:
        coordinates = project_umap(
            latent=latent,
            seed=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
        )

        png_path = (
            output_dir
            / f"{checkpoint_path.stem}_latent_umap.png"
        )
        csv_path = (
            output_dir
            / f"{checkpoint_path.stem}_latent_umap.csv"
        )

        save_plot(
            coordinates=coordinates,
            labels=labels,
            title=f"{title_base} — UMAP",
            x_label="UMAP 1",
            y_label="UMAP 2",
            output_path=png_path,
            dpi=dpi,
        )

        save_coordinates(
            coordinates=coordinates,
            labels=labels,
            method="UMAP",
            output_path=csv_path,
        )

        created.extend([png_path, csv_path])

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return created


def build_parser() -> argparse.ArgumentParser:
    """Définit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualiser l'espace latent des VAE et CVAE."
        )
    )

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pca", "umap"],
        default=["pca", "umap"],
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Valide les arguments."""
    if not 1 <= args.max_samples <= 10_000:
        raise ValueError(
            "max-samples doit être compris entre 1 et 10000."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size doit être strictement positif."
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers ne peut pas être négatif."
        )

    if args.dpi <= 0:
        raise ValueError(
            "dpi doit être strictement positif."
        )

    if args.umap_n_neighbors < 2:
        raise ValueError(
            "umap-n-neighbors doit être supérieur ou égal à 2."
        )

    if not 0.0 <= args.umap_min_dist <= 1.0:
        raise ValueError(
            "umap-min-dist doit être compris entre 0 et 1."
        )

    args.methods = list(dict.fromkeys(args.methods))


def main() -> None:
    """Point d'entrée principal."""
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)
    set_seed(args.seed)

    device = select_device(args.device)
    pin_memory = device.type == "cuda"

    if args.output_dir.is_absolute():
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            PROJECT_ROOT / args.output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = create_test_dataset()

    dataloader = create_latent_loader(
        dataset=dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
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

    print("=" * 72)
    print("VISUALISATION DES ESPACES LATENTS — VAE ET CVAE")
    print("=" * 72)
    print(f"Appareil utilisé      : {device}")
    print(f"Images utilisées      : {len(dataloader.dataset)}")
    print(f"Méthodes              : {', '.join(args.methods)}")
    print(f"Checkpoints           : {len(checkpoint_paths)}")
    print(f"Dossier de sortie     : {output_dir}")
    print("=" * 72)

    all_created: list[Path] = []

    for index, checkpoint_path in enumerate(
        checkpoint_paths,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(checkpoint_paths)}] "
            f"{checkpoint_path.name}"
        )

        created = process_checkpoint(
            checkpoint_path=checkpoint_path,
            dataloader=dataloader,
            device=device,
            methods=args.methods,
            output_dir=output_dir,
            seed=args.seed,
            dpi=args.dpi,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

        all_created.extend(created)

        print(
            f"  -> {len(created)} fichier(s) créé(s)."
        )

    print()
    print("=" * 72)
    print("VISUALISATION TERMINÉE")
    print(f"Nombre de fichiers créés : {len(all_created)}")
    print("=" * 72)

    for path in all_created:
        print(
            f"{path} -> {path.stat().st_size} octets"
        )


if __name__ == "__main__":
    main()