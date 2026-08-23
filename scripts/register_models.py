"""Empaquette et enregistre tous les modeles du catalogue dans le MLflow Model Registry.

Ce script ne reentraine rien : il part des checkpoints deja figes et les
transforme en modeles MLflow versionnes, que l'application charge ensuite via
`mlflow.pyfunc.load_model`.

Chaque modele est empaquete avec :
- ses poids (artefact `checkpoint`) ;
- un descripteur JSON (artefact `spec`) indiquant quel loader de
  `backend.adapters` reconstruit l'adaptateur ;
- son YAML de configuration pour la seule famille MNIST, les deux autres
  embarquant leur configuration dans le checkpoint ;
- le code source minimal necessaire au chargement.

Le code embarque est mis en scene dans un dossier temporaire plutot que
d'etre pris en place : `code_paths` copie l'integralite d'un dossier passe en
argument, et `projects/` pese plus de 100 Mo (resultats, figures, modeles
MLflow de David). Seuls les modules Python reellement importes au chargement
sont recopies, soit environ 470 Ko.

Usage :
    python scripts/register_models.py                      # tout le catalogue
    python scripts/register_models.py --dataset mnist
    python scripts/register_models.py --model mnist/cvae_main
    python scripts/register_models.py --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

from backend.catalog import Catalog, ModelEntry
from backend.mlflow_pyfunc import AdapterPyfuncModel
from backend.mlflow_registry import configure_tracking, registered_name

# Modules importes lorsqu'un adaptateur est reconstruit. `backend/adapters/
# __init__.py` importe les trois familles, donc les trois sont necessaires
# quel que soit le modele empaquete.
STAGED_CODE = [
    Path("backend") / "__init__.py",
    Path("backend") / "adapters",
    Path("backend") / "mlflow_pyfunc.py",
    Path("src"),
    Path("projects") / "david_fashion_mnist" / "models",
    Path("projects") / "blaise_celeba" / "models",
]

PIP_REQUIREMENTS = ["torch", "torchvision", "numpy", "pandas", "pillow", "pyyaml", "tqdm", "mlflow"]


def stage_code(destination: Path) -> List[str]:
    """Recopie les modules necessaires en conservant l'arborescence des imports.

    Les imports sont absolus (`from projects.blaise_celeba.models...`), donc
    la hierarchie doit etre preservee telle quelle sous la racine de mise en
    scene. Les dossiers intermediaires sans `__init__.py` fonctionnent grace
    aux paquets-espaces de noms (PEP 420), comme dans le depot.
    """
    for relative in STAGED_CODE:
        source = ROOT_DIR / relative
        if not source.exists():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(source, target)

    # `code_paths` attend des chemins de premier niveau : MLflow recopie chaque
    # entree sous `code/` en conservant son nom de base.
    return [str(destination / name) for name in ("backend", "src", "projects") if (destination / name).exists()]


def build_spec(entry: ModelEntry) -> dict:
    """Descripteur JSON embarque avec le modele, lu par AdapterPyfuncModel."""
    return {
        "model_id": entry.model_id,
        "dataset_id": entry.dataset_id,
        "loader": entry.loader,
        "label": entry.label,
        "beta": entry.beta,
        "conditional": entry.conditional,
        "family": entry.family,
        "ablation_series": entry.ablation_series,
    }


# Signature declaree explicitement plutot qu'inferee depuis un exemple.
# L'inference typait "n" et "seed" en entiers requis, or ils sont facultatifs :
# MLflow refusait alors toute requete les laissant a null, avec
# "Can not safely convert object to int64". Les colonnes numeriques
# facultatives sont donc des doubles, seul type numerique capable de porter
# une valeur manquante, et "z" voyage en JSON pour rester servable en HTTP.
INPUT_SCHEMA = Schema(
    [
        ColSpec(DataType.string, "op"),
        ColSpec(DataType.string, "z", required=False),
        ColSpec(DataType.double, "class_label", required=False),
        ColSpec(DataType.string, "image_base64", required=False),
        ColSpec(DataType.double, "n", required=False),
        ColSpec(DataType.double, "seed", required=False),
    ]
)
MODEL_SIGNATURE = ModelSignature(inputs=INPUT_SCHEMA)


def input_example_for(entry: ModelEntry) -> pd.DataFrame:
    """Exemple d'appel joint au modele, visible dans l'interface MLflow."""
    return pd.DataFrame(
        [
            {
                "op": "sample",
                "z": None,
                "class_label": 0.0 if entry.conditional else None,
                "image_base64": None,
                "n": 1.0,
                "seed": 42.0,
            }
        ]
    )


def register_entry(entry: ModelEntry, code_paths: List[str], catalog: Catalog) -> Optional[str]:
    """Empaquette un modele et l'enregistre. Renvoie la version creee."""
    name = registered_name(entry.model_id)

    with tempfile.TemporaryDirectory() as spec_dir:
        spec_path = Path(spec_dir) / "spec.json"
        spec_path.write_text(json.dumps(build_spec(entry), indent=2), encoding="utf-8")

        artifacts = {
            "checkpoint": str(entry.checkpoint_path),
            "spec": str(spec_path),
        }
        if entry.config_path is not None:
            artifacts["config"] = str(entry.config_path)

        with mlflow.start_run(run_name=f"package-{name}"):
            mlflow.set_tags(
                {
                    "model_id": entry.model_id,
                    "dataset": entry.dataset_id,
                    "family": entry.family,
                    "conditional": str(entry.conditional),
                    "loader": entry.loader,
                }
            )
            if entry.beta is not None:
                mlflow.log_param("beta", entry.beta)
            mlflow.log_param("checkpoint", entry.checkpoint_path.name)

            # Les metriques de l'adaptateur charge decrivent le modele empaquete.
            adapter = catalog.adapter(entry.model_id)
            mlflow.log_param("latent_dim", adapter.latent_dim)
            mlflow.log_param("num_conditions", adapter.num_conditions)
            mlflow.log_param("output_range", str(adapter.output_range))

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AdapterPyfuncModel(),
                artifacts=artifacts,
                code_paths=code_paths,
                signature=MODEL_SIGNATURE,
                input_example=input_example_for(entry),
                pip_requirements=PIP_REQUIREMENTS,
                registered_model_name=name,
            )

    from mlflow.tracking import MlflowClient

    versions = MlflowClient().search_model_versions(f"name='{name}'")
    return str(max(int(version.version) for version in versions)) if versions else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", help="Ne traiter qu'un dataset (mnist, fashion_mnist, celeba)")
    parser.add_argument("--model", help="Ne traiter qu'un modele, par son identifiant de catalogue")
    parser.add_argument("--list", action="store_true", help="Lister les modeles sans rien enregistrer")
    args = parser.parse_args()

    catalog = Catalog(device="cpu")

    try:
        entries = catalog.models(args.dataset)
    except KeyError as error:
        parser.error(str(error))
        return

    if args.model:
        entries = [entry for entry in entries if entry.model_id == args.model]
        if not entries:
            parser.error(f"Modele inconnu ou indisponible : {args.model}")

    if not entries:
        print("Aucun checkpoint disponible. Rien a enregistrer.")
        return

    if args.list:
        print(f"{len(entries)} modele(s) disponibles :")
        for entry in entries:
            print(f"  {entry.model_id:40s} -> {registered_name(entry.model_id)}")
        return

    uri = configure_tracking()
    print(f"Tracking MLflow : {uri}")
    print(f"{len(entries)} modele(s) a empaqueter.\n")

    with tempfile.TemporaryDirectory() as staging:
        code_paths = stage_code(Path(staging))
        for entry in entries:
            version = register_entry(entry, code_paths, catalog)
            print(f"  {entry.model_id:40s} -> {registered_name(entry.model_id)} v{version}")

    print("\nTermine. Inspecter le registre avec :")
    print(f"  mlflow ui --backend-store-uri {uri}")


if __name__ == "__main__":
    main()
