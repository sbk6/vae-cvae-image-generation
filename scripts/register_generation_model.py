"""Enregistre dans le Model Registry MLflow un service de génération unique
couvrant tous les datasets de l'équipe qui ont déjà un CVAE (et, si
disponible, un VAE + ses centroïdes latents) entraînés — voir
configs/deployment_registry.yaml. Un seul modèle enregistré = un seul
serveur `mlflow models serve` = un seul port pour toute l'application web,
avec deux actions possibles : "generate" (CVAE) et "interpolate" (VAE).

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
from src.serving.generation_pyfunc import ImageGenerationService, build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default="configs/deployment_registry.yaml")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", type=str, default="vae-cvae-mnist-deployment")
    parser.add_argument("--registered-model-name", type=str, default="image_generator")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    artifacts = build_artifacts(args.registry)
    generate_datasets = sorted(k[len("cvae_checkpoint_"):] for k in artifacts if k.startswith("cvae_checkpoint_"))
    interpolate_datasets = sorted(k[len("vae_checkpoint_"):] for k in artifacts if k.startswith("vae_checkpoint_"))

    input_example = pd.DataFrame({"action": ["generate"], "dataset": ["mnist"], "classe": [7.0]})
    # Tous les champs sont optionnels dans le schéma : la validation du "bon"
    # jeu de champs pour chaque action est faite dans predict() lui-même
    # (generate a besoin de "classe", interpolate de "classe_a"/"classe_b"/"t").
    signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", "action", required=False),
                ColSpec("string", "dataset", required=False),
                ColSpec("double", "classe", required=False),
                ColSpec("double", "classe_a", required=False),
                ColSpec("double", "classe_b", required=False),
                ColSpec("double", "t", required=False),
            ]
        ),
        outputs=Schema([ColSpec("string", "image_base64")]),
    )

    with mlflow.start_run(run_name="register_image_generator"):
        mlflow.set_tags(
            {
                "purpose": "deployment",
                "generate_datasets": ",".join(generate_datasets),
                "interpolate_datasets": ",".join(interpolate_datasets),
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="image_generator",
            python_model=ImageGenerationService(),
            artifacts=artifacts,
            code_paths=[str(ROOT_DIR / "src")],
            signature=signature,
            input_example=input_example,
            registered_model_name=args.registered_model_name,
            pip_requirements=["torch", "pillow", "numpy", "pandas", "pyyaml", "mlflow"],
        )
        print(f"Modèle enregistré : {model_info.model_uri}")
        print(f"Datasets disponibles pour 'generate' : {generate_datasets}")
        print(f"Datasets disponibles pour 'interpolate' : {interpolate_datasets}")

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{args.registered_model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    print(f"Dernière version enregistrée : {args.registered_model_name} v{latest.version}")
    print(f"URI utilisable pour servir : models:/{args.registered_model_name}/{latest.version}")


if __name__ == "__main__":
    main()
