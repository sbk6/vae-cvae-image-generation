"""Acces au MLflow Model Registry partage par l'application.

Depuis l'integration MLflow, l'API ne charge plus aucun modele en important
directement un adaptateur : tout passe par
`mlflow.pyfunc.load_model("models:/<nom>/<version>")`. Le catalogue ne sert
plus qu'a decrire ce qui existe et a savoir quoi enregistrer ; c'est le
Registry qui fournit les modeles a l'execution.

Deux stores MLflow coexistent volontairement dans le depot :

- `projects/david_fashion_mnist/mlflow.db` — celui de David, restreint a
  Fashion-MNIST, servi en HTTP par `mlflow models serve`. C'est son livrable,
  laisse intact ;
- `mlflow.db` a la racine — celui de l'application, qui couvre les trois
  datasets et ajoute l'operation `encode`, indispensable a la reconstruction
  et a l'interpolation depuis de vraies images.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parent.parent

# Emplacement du store MLflow. La base et les artefacts doivent vivre dans un
# meme dossier pour pouvoir etre montes sur un unique volume : monter les
# artefacts sans la base laisserait un registre qui referme des chemins
# existants mais ne connait plus aucun modele.
#
# En local, les deux restent a la racine du depot pour ne rien changer aux
# habitudes ; en conteneur, MLFLOW_STORE_DIR les regroupe (voir Dockerfile).
_STORE_DIR = os.environ.get("MLFLOW_STORE_DIR")
MLFLOW_STORE_DIR = Path(_STORE_DIR) if _STORE_DIR else ROOT_DIR

MLFLOW_DB_PATH = MLFLOW_STORE_DIR / "mlflow.db"
MLFLOW_ARTIFACT_DIR = MLFLOW_STORE_DIR / "mlartifacts"
EXPERIMENT_NAME = "vae_demo_app"


def tracking_uri() -> str:
    """URI du store de l'application. `as_posix` evite les antislashs Windows."""
    return f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"


def configure_tracking() -> str:
    """Pointe MLflow vers le store de l'application et garantit l'experience."""
    uri = tracking_uri()
    MLFLOW_STORE_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def registered_name(model_id: str) -> str:
    """Traduit un identifiant de catalogue en nom de modele enregistre.

    `models:/<nom>/<version>` reserve la barre oblique : les identifiants du
    catalogue (`mnist/cvae_main`) la remplacent donc par un tiret. La
    transformation reste reversible et regroupe les modeles par dataset dans
    l'interface du Registry.
    """
    return model_id.replace("/", "-")


def is_registry_available() -> bool:
    """Vrai si le store existe deja, sans le creer au passage.

    Interroger MLflow sur une base absente la creerait vide, ce qui masquerait
    le fait que l'enregistrement n'a jamais ete lance.
    """
    return MLFLOW_DB_PATH.exists()


class RegistryGateway:
    """Charge et met en cache les modeles pyfunc du Registry.

    Un modele pyfunc est immuable une fois charge, et son chargement coute une
    lecture de checkpoint : on le garde en memoire. Thread-safe, l'API etant
    servie en multi-thread.
    """

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}
        self._versions: Dict[str, str] = {}
        self._descriptions: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()
        self._configured = False

    def _ensure_configured(self) -> None:
        if not self._configured:
            configure_tracking()
            self._configured = True

    def available_names(self) -> List[str]:
        """Noms enregistres dans le Registry, ou liste vide s'il n'existe pas."""
        if not is_registry_available():
            return []
        self._ensure_configured()
        try:
            return sorted(model.name for model in MlflowClient().search_registered_models())
        except Exception:
            return []

    def latest_version(self, name: str) -> Optional[str]:
        """Derniere version enregistree d'un modele, ou None s'il est absent."""
        self._ensure_configured()
        try:
            versions = MlflowClient().search_model_versions(f"name='{name}'")
        except Exception:
            return None
        if not versions:
            return None
        return str(max(int(version.version) for version in versions))

    def load(self, model_id: str):
        """Renvoie le modele pyfunc correspondant, en le chargeant au premier appel."""
        if model_id in self._models:
            return self._models[model_id]

        with self._lock:
            # Un autre thread a pu charger le modele pendant l'attente du verrou.
            if model_id in self._models:
                return self._models[model_id]

            if not is_registry_available():
                raise FileNotFoundError(
                    "Aucun Model Registry trouve. Enregistrer les modeles avec : "
                    "python scripts/register_models.py"
                )

            self._ensure_configured()
            name = registered_name(model_id)
            version = self.latest_version(name)
            if version is None:
                raise KeyError(
                    f"Modele '{model_id}' absent du Registry (nom attendu : '{name}'). "
                    "Relancer : python scripts/register_models.py"
                )

            model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
            self._models[model_id] = model
            self._versions[model_id] = version
            return model

    def describe(self, model_id: str) -> Dict[str, object]:
        """Metadonnees d'un modele, lues dans le Registry et non dans l'adaptateur.

        `latent_dim`, `num_conditions` et `output_range` sont journalises comme
        parametres du run d'empaquetage par `scripts/register_models.py`. Les
        lire ici evite a l'API de charger le modele juste pour dimensionner les
        curseurs latents ou valider un vecteur z, et respecte le principe que
        seul MLflow expose les modeles.
        """
        if model_id in self._descriptions:
            return self._descriptions[model_id]

        self._ensure_configured()
        name = registered_name(model_id)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise KeyError(f"Modele '{model_id}' absent du Registry (nom attendu : '{name}').")

        latest = max(versions, key=lambda version: int(version.version))
        params = client.get_run(latest.run_id).data.params
        tags = client.get_run(latest.run_id).data.tags

        def as_int(key: str) -> Optional[int]:
            value = params.get(key)
            return int(value) if value not in (None, "", "None") else None

        description = {
            "registered_name": name,
            "version": str(latest.version),
            "latent_dim": as_int("latent_dim"),
            "num_conditions": as_int("num_conditions"),
            "conditional": tags.get("conditional") == "True",
            "beta": float(params["beta"]) if params.get("beta") not in (None, "", "None") else None,
            "output_range": params.get("output_range"),
        }
        self._descriptions[model_id] = description
        return description

    def version_of(self, model_id: str) -> Optional[str]:
        """Version chargee pour ce modele, si elle l'a deja ete."""
        return self._versions.get(model_id)

    def loaded_ids(self) -> List[str]:
        return sorted(self._models)
