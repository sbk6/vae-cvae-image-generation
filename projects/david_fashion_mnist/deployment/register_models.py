"""
Enregistre les checkpoints finaux VAE et CVAE dans le MLflow Model
Registry, en réutilisant les wrappers définis dans mlflow_models.py.

Important
---------
Ce script NE réentraîne rien. Il part des checkpoints déjà figés :

    checkpoints/vae_beta_1_seed42_final.pt
    checkpoints/cvae_beta_1_seed42_final.pt

et les transforme en "MLflow Models" versionnés, prêts à être servis
avec `mlflow models serve`.

Utilisation
-----------
    python -m deployment.register_models

    python -m deployment.register_models \
        --vae-checkpoint checkpoints/vae_beta_1_seed42_final.pt \
        --cvae-checkpoint checkpoints/cvae_beta_1_seed42_final.pt

Ce que fait ce script pour chaque modèle
-----------------------------------------
1. mlflow.pyfunc.log_model(...)   -> crée le "MLflow Model" packagé
                                      dans un run MLflow (mlartifacts/).
2. mlflow.register_model(...)     -> l'enregistre dans le Model
                                      Registry sous un nom stable
                                      (ex: "FashionMNIST-CVAE").

À la fin, le Model Registry contient :

    FashionMNIST-VAE   version 1
    FashionMNIST-CVAE  version 1

Ces noms de modèle sont ensuite utilisés pour servir :

    mlflow models serve -m "models:/FashionMNIST-VAE/1"  -p 5001
    mlflow models serve -m "models:/FashionMNIST-CVAE/1" -p 5002
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch

from deployment.mlflow_models import CVAEPyfuncModel, VAEPyfuncModel

# ================================================================
# CHEMINS -- identiques à training/train_vae.py, pour utiliser
# exactement le même backend MLflow (même mlflow.db, même
# mlartifacts/).
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

DEFAULT_VAE_CHECKPOINT = CHECKPOINT_DIR / "vae_beta_1_seed42_final.pt"
DEFAULT_CVAE_CHECKPOINT = CHECKPOINT_DIR / "cvae_beta_1_seed42_final.pt"

DEPLOYMENT_EXPERIMENT_NAME = "fashion_mnist_deployment"

VAE_REGISTERED_NAME = "FashionMNIST-VAE"
CVAE_REGISTERED_NAME = "FashionMNIST-CVAE"

# Dossiers de code à embarquer dans le modèle MLflow, pour que
# VAE / CVAE / mlflow_models.py restent importables au chargement,
# y compris sur une autre machine (celle de votre collègue).
CODE_PATHS = [
    str(PROJECT_ROOT / "models"),
    str(PROJECT_ROOT / "deployment"),
]


def configure_mlflow_tracking() -> str:
    """
    Configure MLflow pour utiliser exactement le même backend que
    training/train_vae.py (SQLite local + mlartifacts/).
    """

    database_path = MLFLOW_DB_PATH.resolve().as_posix()
    tracking_uri = f"sqlite:///{database_path}"

    mlflow.set_tracking_uri(tracking_uri)

    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    experiment = mlflow.get_experiment_by_name(DEPLOYMENT_EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=DEPLOYMENT_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
        )

    mlflow.set_experiment(DEPLOYMENT_EXPERIMENT_NAME)

    return tracking_uri


def register_vae(checkpoint_path: Path) -> None:
    """
    Empaquette et enregistre le VAE dans le Model Registry.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint VAE introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    configuration = checkpoint["configuration"]

    # Un petit exemple d'entrée, utile pour que MLflow documente
    # automatiquement le format attendu par l'endpoint.
    input_example = pd.DataFrame({"z": [None]})

    with mlflow.start_run(run_name="register_vae"):

        mlflow.log_params(
            {
                "source_checkpoint": checkpoint_path.name,
                "beta": checkpoint.get("beta"),
                "training_seed": configuration.get("training_seed"),
                "split_seed": configuration.get("split_seed"),
                "latent_dim": configuration["latent_dim"],
                "hidden_dim": configuration["hidden_dim"],
            }
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=VAEPyfuncModel(),
            artifacts={"checkpoint": str(checkpoint_path)},
            code_paths=CODE_PATHS,
            input_example=input_example,
            pip_requirements=[
                "torch",
                "numpy",
                "pandas",
                "pillow",
                "mlflow",
            ],
        )

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=VAE_REGISTERED_NAME,
        )

        print(
            f"VAE enregistré : {VAE_REGISTERED_NAME} "
            f"version {registered_model.version}"
        )


def register_cvae(checkpoint_path: Path) -> None:
    """
    Empaquette et enregistre le CVAE dans le Model Registry.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint CVAE introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    configuration = checkpoint["configuration"]

    input_example = pd.DataFrame({"class": [7], "z": [None]})

    with mlflow.start_run(run_name="register_cvae"):

        mlflow.log_params(
            {
                "source_checkpoint": checkpoint_path.name,
                "beta": checkpoint.get("beta"),
                "training_seed": configuration.get("training_seed"),
                "split_seed": configuration.get("split_seed"),
                "latent_dim": configuration["latent_dim"],
                "hidden_dim": configuration["hidden_dim"],
            }
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CVAEPyfuncModel(),
            artifacts={"checkpoint": str(checkpoint_path)},
            code_paths=CODE_PATHS,
            input_example=input_example,
            pip_requirements=[
                "torch",
                "numpy",
                "pandas",
                "pillow",
                "mlflow",
            ],
        )

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=CVAE_REGISTERED_NAME,
        )

        print(
            f"CVAE enregistré : {CVAE_REGISTERED_NAME} "
            f"version {registered_model.version}"
        )


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Enregistre les checkpoints VAE/CVAE finaux dans le "
            "MLflow Model Registry."
        )
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=DEFAULT_VAE_CHECKPOINT,
        help="Chemin du checkpoint VAE figé.",
    )

    parser.add_argument(
        "--cvae-checkpoint",
        type=Path,
        default=DEFAULT_CVAE_CHECKPOINT,
        help="Chemin du checkpoint CVAE figé.",
    )

    return parser


def main() -> None:

    parser = build_argument_parser()
    args = parser.parse_args()

    tracking_uri = configure_mlflow_tracking()

    print("=" * 74)
    print("ENREGISTREMENT DES MODÈLES DANS LE MLFLOW MODEL REGISTRY")
    print("=" * 74)
    print(f"Tracking URI      : {tracking_uri}")
    print(f"Checkpoint VAE    : {args.vae_checkpoint}")
    print(f"Checkpoint CVAE   : {args.cvae_checkpoint}")
    print("=" * 74)

    register_vae(args.vae_checkpoint)
    register_cvae(args.cvae_checkpoint)

    print("=" * 74)
    print("TERMINÉ")
    print("=" * 74)
    print("Pour servir les modèles, dans deux terminaux séparés :")
    print()
    print(
        f'  mlflow models serve -m "models:/{VAE_REGISTERED_NAME}/1" '
        "-p 5001 --env-manager local"
    )
    print(
        f'  mlflow models serve -m "models:/{CVAE_REGISTERED_NAME}/1" '
        "-p 5002 --env-manager local"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()