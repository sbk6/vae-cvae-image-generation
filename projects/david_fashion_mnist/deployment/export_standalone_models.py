"""
Exporte les modèles MLflow déjà enregistrés (FashionMNIST-VAE,
FashionMNIST-CVAE) vers des dossiers autonomes et portables.

Pourquoi ce script ?
---------------------
`register_models.py` enregistre les modèles dans le Model Registry
LOCAL (mlflow.db + mlartifacts/), qui ne sont pas versionnés dans
Git et n'existent que sur votre machine.

Si votre collègue travaille de son côté et récupère le projet via
`git pull`, il n'a ni votre mlflow.db, ni vos serveurs `mlflow models
serve` qui tournent chez vous. Il a seulement les fichiers que vous
lui transmettez.

Ce script résout ça : il copie le modèle MLflow packagé (celui
généré par register_models.py) dans un dossier autonome, qui
contient tout ce qu'il faut pour être servi ailleurs, sans registre
ni connexion à votre machine :

    deployment/packaged_models/
    ├── vae_model/
    │   ├── MLmodel
    │   ├── python_model.pkl
    │   ├── artifacts/checkpoint          (le .pt)
    │   ├── code/                         (models/, deployment/)
    │   └── requirements.txt
    └── cvae_model/
        └── (même structure)

Une fois ce dossier transmis (git, zip, Drive -- selon la taille
et vos habitudes d'équipe), votre collègue lance, SUR SA PROPRE
MACHINE, sans rien installer d'autre que mlflow/torch :

    mlflow models serve -m deployment/packaged_models/vae_model  -p 5001 --env-manager local
    mlflow models serve -m deployment/packaged_models/cvae_model -p 5002 --env-manager local

Utilisation
-----------
    python -m deployment.export_standalone_models

Prérequis : avoir déjà lancé register_models.py au moins une fois
(les modèles doivent exister dans le Model Registry local).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import mlflow.artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPORT_DIR = PROJECT_ROOT / "deployment" / "packaged_models"

VAE_MODEL_URI = "models:/FashionMNIST-VAE/1"
CVAE_MODEL_URI = "models:/FashionMNIST-CVAE/1"


def configure_mlflow_tracking() -> None:
    """
    Pointe vers le même backend MLflow que register_models.py, pour
    retrouver les modèles déjà enregistrés.
    """

    database_path = MLFLOW_DB_PATH.resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")


def export_model(model_uri: str, destination: Path) -> None:
    """
    Télécharge/copie un modèle MLflow enregistré vers un dossier
    autonome.
    """

    if destination.exists():
        print(f"  Dossier existant supprimé : {destination}")
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(destination),
    )

    print(f"  Exporté vers : {destination}")


def main() -> None:

    configure_mlflow_tracking()

    print("=" * 74)
    print("EXPORT DES MODÈLES VERS DES DOSSIERS PORTABLES")
    print("=" * 74)

    print("Export du VAE ...")
    export_model(VAE_MODEL_URI, EXPORT_DIR / "vae_model")

    print("Export du CVAE ...")
    export_model(CVAE_MODEL_URI, EXPORT_DIR / "cvae_model")

    print("=" * 74)
    print("TERMINÉ")
    print("=" * 74)
    print(f"Dossiers créés dans : {EXPORT_DIR}")
    print()
    print("À transmettre à votre collègue (git ou zip selon la taille).")
    print()
    print("Sur sa machine, il devra lancer :")
    print()
    print(
        "  mlflow models serve -m deployment/packaged_models/vae_model "
        "-p 5001 --env-manager local"
    )
    print(
        "  mlflow models serve -m deployment/packaged_models/cvae_model "
        "-p 5002 --env-manager local"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()