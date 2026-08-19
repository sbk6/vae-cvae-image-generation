"""Wrapper MLflow pyfunc pour le CVAE : expose une fonction predict() simple
(classe demandée -> image PNG encodée en base64) afin que le modèle puisse
être servi par `mlflow models serve` sans que l'appelant ait besoin de
connaître PyTorch ni l'architecture du réseau.

Contrat d'API (voir docs/DEPLOIEMENT.md pour le détail complet) :
- entrée : une ligne par image demandée, colonne "classe" (entier 0-9)
- sortie : une chaîne par ligne, l'image PNG encodée en base64
"""
import base64
import io

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

        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model = CVAE(
            channels=config["dataset"]["channels"],
            image_size=tuple(config["dataset"]["image_size"]),
            latent_dim=config["model"]["latent_dim"],
            num_conditions=config["dataset"]["num_classes"],
            condition_mode=config["dataset"].get("condition_mode", "one_hot"),
            hidden_channels=config["model"]["hidden_channels"],
        )
        checkpoint = torch.load(context.artifacts["checkpoint"], map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_classes = config["dataset"]["num_classes"]

    def _generate_one(self, classe: int) -> str:
        if not (0 <= classe < self.num_classes):
            raise ValueError(f"classe doit être comprise entre 0 et {self.num_classes - 1}, reçu {classe}")

        with torch.no_grad():
            sample = self.model.sample(classe, n=1)[0, 0]  # (H, W), valeurs dans [-1, 1]

        pixels = ((sample.numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        classes = model_input["classe"].astype(int).tolist()
        images = [self._generate_one(c) for c in classes]
        return pd.Series(images, name="image_base64")
