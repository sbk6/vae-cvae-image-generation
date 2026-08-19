"""Enregistre dans le Model Registry MLflow un service CVAE unique couvrant
tous les datasets de l'équipe qui ont déjà un checkpoint entraîné (voir
configs/deployment_registry.yaml). Un seul modèle enregistré = un seul
serveur `mlflow models serve` = un seul port pour toute l'application web,
quel que soit le nombre de datasets déjà disponibles.

Voir docs/DEPLOIEMENT.md pour le contrat d'API complet destiné à l'équipe
qui construit l'application web.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from src.serving.cvae_pyfunc import CVAEGenerator, build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default="configs/deployment_registry.yaml")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", type=str, default="vae-cvae-mnist-deployment")
    parser.add_argument("--registered-model-name", type=str, default="cvae_generator")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    artifacts = build_artifacts(args.registry)
    available_datasets = sorted(k[len("checkpoint_"):] for k in artifacts if k.startswith("checkpoint_"))

    input_example = pd.DataFrame({"dataset": ["mnist"], "classe": [7]})
    # "dataset" est marqué explicitement optionnel (défaut "mnist" géré dans predict()) :
    # infer_signature() aurait déduit un champ obligatoire à partir de l'exemple.
    signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", "dataset", required=False),
                ColSpec("long", "classe", required=True),
            ]
        ),
        outputs=Schema([ColSpec("string", "image_base64")]),
    )

    with mlflow.start_run(run_name="register_cvae_generator"):
        mlflow.set_tags({"purpose": "deployment", "model": "cvae", "datasets": ",".join(available_datasets)})
        model_info = mlflow.pyfunc.log_model(
            name="cvae_generator",
            python_model=CVAEGenerator(),
            artifacts=artifacts,
            code_paths=[str(ROOT_DIR / "src")],
            signature=signature,
            input_example=input_example,
            registered_model_name=args.registered_model_name,
            pip_requirements=["torch", "pillow", "numpy", "pandas", "pyyaml", "mlflow"],
        )
        print(f"Modèle enregistré : {model_info.model_uri}")
        print(f"Datasets inclus dans cette version : {available_datasets}")

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{args.registered_model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    print(f"Dernière version enregistrée : {args.registered_model_name} v{latest.version}")
    print(f"URI utilisable pour servir : models:/{args.registered_model_name}/{latest.version}")


if __name__ == "__main__":
    main()
