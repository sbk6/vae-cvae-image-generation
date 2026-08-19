"""Wrapper MLflow pyfunc exposant, sous UN SEUL endpoint, le CVAE de chaque
dataset de l'équipe (MNIST aujourd'hui, Fashion-MNIST et CelebA à venir).

Plutôt que de démarrer un serveur MLflow par dataset (donc un port par
dataset), ce wrapper charge tous les modèles disponibles au démarrage et
route chaque requête vers le bon modèle selon le champ "dataset" reçu en
entrée. Cela donne une seule URL, un seul port, un seul contrat d'API pour
l'application web, qui reste valable quand un nouveau dataset est ajouté.

Contrat d'API (voir docs/DEPLOIEMENT.md pour le détail complet) :
- entrée : une ligne par image demandée, colonnes "classe" (entier) et
  "dataset" (texte, optionnel, "mnist" par défaut)
- sortie : une chaîne par ligne, l'image PNG encodée en base64
"""
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

import mlflow.pyfunc


class CVAEGenerator(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        # `code_paths=["src"]` (voir scripts/register_cvae_model.py) rend le
        # package `src` importable ici, exactement comme dans le reste du dépôt.
        from src.models.cvae import CVAE

        with open(context.artifacts["registry"], "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)["datasets"]

        self.models = {}
        self.num_classes = {}
        skipped = []

        for name, entry in registry.items():
            checkpoint_key = f"checkpoint_{name}"
            config_key = f"config_{name}"
            if checkpoint_key not in context.artifacts or config_key not in context.artifacts:
                skipped.append(name)
                continue

            with open(context.artifacts[config_key], "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            model = CVAE(
                channels=config["dataset"]["channels"],
                image_size=tuple(config["dataset"]["image_size"]),
                latent_dim=config["model"]["latent_dim"],
                num_conditions=config["dataset"]["num_classes"],
                condition_mode=config["dataset"].get("condition_mode", "one_hot"),
                hidden_channels=config["model"]["hidden_channels"],
            )
            checkpoint = torch.load(context.artifacts[checkpoint_key], map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.models[name] = model
            self.num_classes[name] = config["dataset"]["num_classes"]

        if not self.models:
            raise RuntimeError(
                "Aucun modèle CVAE disponible : vérifier configs/deployment_registry.yaml "
                "et l'existence des checkpoints référencés."
            )
        if skipped:
            print(f"[cvae_pyfunc] datasets non chargés (checkpoint absent) : {skipped}")
        print(f"[cvae_pyfunc] datasets disponibles : {sorted(self.models.keys())}")

    def _generate_one(self, dataset: str, classe: int) -> str:
        if dataset not in self.models:
            available = ", ".join(sorted(self.models.keys()))
            raise ValueError(f"dataset '{dataset}' indisponible. Datasets chargés actuellement : {available}")

        num_classes = self.num_classes[dataset]
        if not (0 <= classe < num_classes):
            raise ValueError(f"classe doit être comprise entre 0 et {num_classes - 1} pour '{dataset}', reçu {classe}")

        with torch.no_grad():
            sample = self.models[dataset].sample(classe, n=1)[0]  # (C, H, W), valeurs dans [-1, 1]

        pixels = ((sample.numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        if pixels.shape[0] == 1:
            image = Image.fromarray(pixels[0], mode="L")
        else:
            image = Image.fromarray(np.transpose(pixels, (1, 2, 0)), mode="RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        classes = model_input["classe"].astype(int).tolist()
        if "dataset" in model_input.columns:
            datasets = model_input["dataset"].fillna("mnist").tolist()
        else:
            datasets = ["mnist"] * len(classes)

        images = [self._generate_one(d, c) for d, c in zip(datasets, classes)]
        return pd.Series(images, name="image_base64")


def build_artifacts(registry_path: str) -> dict:
    """Construit le dict d'artefacts à passer à `mlflow.pyfunc.log_model`,
    en n'incluant que les datasets dont le checkpoint existe réellement.
    """
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)["datasets"]

    artifacts = {"registry": registry_path}
    available, missing = [], []
    for name, entry in registry.items():
        checkpoint_path = Path(entry["checkpoint"])
        config_path = Path(entry["config"])
        if checkpoint_path.exists() and config_path.exists():
            artifacts[f"checkpoint_{name}"] = str(checkpoint_path)
            artifacts[f"config_{name}"] = str(config_path)
            available.append(name)
        else:
            missing.append(name)

    if not available:
        raise RuntimeError("Aucun dataset avec un checkpoint disponible dans le registre.")
    print(f"Datasets inclus dans ce déploiement : {available}")
    if missing:
        print(f"Datasets pas encore disponibles (ignorés) : {missing}")
    return artifacts
