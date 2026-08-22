"""
Wrappers MLflow pyfunc pour le VAE et le CVAE Fashion-MNIST.

Ce fichier NE réécrit PAS l'architecture des modèles.

Il réutilise exactement les classes existantes :

    models/vae.py   -> VAE
    models/cvae.py  -> CVAE

Rôle de ce fichier
-------------------
Transformer un checkpoint PyTorch déjà entraîné (.pt) en un objet
que MLflow sait :

1. sauvegarder comme "MLflow Model" (mlflow.pyfunc.log_model) ;
2. enregistrer dans le Model Registry ;
3. servir via `mlflow models serve`, avec un endpoint HTTP standard
   `/invocations`.

Ce fichier ne charge PAS de checkpoint tout seul. C'est
`register_models.py` qui indique le chemin du checkpoint au moment
de l'enregistrement (voir `artifacts={"checkpoint": ...}`).

Format d'entrée attendu par les endpoints (une fois servis)
-------------------------------------------------------------
Ce sont des requêtes JSON envoyées à `/invocations`, au format
MLflow "dataframe_split" (le format standard pour pyfunc).

VAE :

    POST http://localhost:5001/invocations
    {
        "dataframe_split": {
            "columns": ["z"],
            "data": [[null]]
        }
    }

    -> z = null : le serveur tire un vecteur latent aléatoire.
    -> z = [0.12, -0.5, ..., 0.03] (16 valeurs) : z imposé, utile
       pour le slider d'interpolation de l'application web.

CVAE :

    POST http://localhost:5002/invocations
    {
        "dataframe_split": {
            "columns": ["class", "z"],
            "data": [[7, null]]
        }
    }

    -> "class" : entier entre 0 et 9 (obligatoire pour le CVAE).
    -> "z" : comme pour le VAE, optionnel.

Plusieurs lignes peuvent être envoyées dans "data" pour générer
plusieurs images en une seule requête.

Format de sortie
-----------------
Une liste JSON, une entrée par ligne d'entrée :

    [
        {
            "image_base64": "iVBORw0KGgoAAAANS...",
            "image_format": "png",
            "width": 28,
            "height": 28,
            "requested_class": 7          # uniquement pour le CVAE
        },
        ...
    ]

`image_base64` est directement affichable côté application web avec :

    <img src="data:image/png;base64,{image_base64}" />
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ces imports fonctionnent car "models" est inclus dans code_paths
# lors de l'enregistrement (voir register_models.py). MLflow copie
# le dossier "models/" à l'intérieur du modèle packagé.
from models.vae import VAE
from models.cvae import CVAE


# ================================================================
# OUTILS PARTAGÉS
# ================================================================


def _tensor_image_to_base64_png(image_tensor: torch.Tensor) -> str:
    """
    Convertit une image générée (valeurs entre 0 et 1) en PNG base64.

    Parameters
    ----------
    image_tensor:
        Tenseur de forme [1, 28, 28] ou [28, 28], valeurs dans [0, 1].
    """

    array = image_tensor.detach().cpu().numpy()

    # Retire la dimension du canal si nécessaire : [1, 28, 28] -> [28, 28].
    if array.ndim == 3:
        array = array[0]

    # Les pixels du modèle sont dans [0, 1] (sortie Sigmoid).
    # Une image PNG classique attend des entiers entre 0 et 255.
    pixels_uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(pixels_uint8, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return encoded


def _parse_optional_latent_vector(
    raw_value: Any,
    latent_dim: int,
) -> Optional[torch.Tensor]:
    """
    Transforme la valeur reçue dans la colonne "z" en tenseur PyTorch,
    ou renvoie None si aucun z n'a été fourni (auquel cas l'appelant
    doit en tirer un aléatoirement).

    Accepte :
    - None / NaN -> None ;
    - une liste ou un tableau numpy de "latent_dim" valeurs.
    """

    if raw_value is None:
        return None

    # pandas peut transformer un JSON "null" en NaN flottant.
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return None

    latent_values = np.asarray(raw_value, dtype=np.float32)

    if latent_values.shape != (latent_dim,):
        raise ValueError(
            "Le vecteur latent 'z' doit contenir exactement "
            f"{latent_dim} valeurs. "
            f"Valeurs reçues : {latent_values.shape}."
        )

    return torch.from_numpy(latent_values).unsqueeze(0)


# ================================================================
# WRAPPER VAE
# ================================================================


class VAEPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    Modèle MLflow pyfunc encapsulant le VAE Fashion-MNIST.

    Ce wrapper ne contient aucune logique de modèle : il se contente
    de charger la classe VAE existante et de traduire les requêtes
    HTTP JSON en appels à `model.decode(z)`.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Appelé automatiquement par MLflow au chargement du modèle.

        Charge le checkpoint .pt fourni comme artefact lors de
        l'enregistrement (voir register_models.py).
        """

        checkpoint_path = context.artifacts["checkpoint"]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        configuration = checkpoint["configuration"]

        self.model = VAE(
            latent_dim=configuration["latent_dim"],
            hidden_dim=configuration["hidden_dim"],
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.latent_dim = configuration["latent_dim"]

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """
        Génère une image par ligne reçue.

        model_input doit contenir une colonne "z" (optionnelle).
        """

        if "z" not in model_input.columns:
            model_input = model_input.copy()
            model_input["z"] = None

        results: list[dict] = []

        for _, row in model_input.iterrows():

            z = _parse_optional_latent_vector(
                raw_value=row["z"],
                latent_dim=self.latent_dim,
            )

            if z is None:
                z = torch.randn(1, self.latent_dim)

            with torch.no_grad():
                generated_image = self.model.decode(z)

            results.append(
                {
                    "image_base64": _tensor_image_to_base64_png(
                        generated_image[0]
                    ),
                    "image_format": "png",
                    "width": 28,
                    "height": 28,
                }
            )

        return results


# ================================================================
# WRAPPER CVAE
# ================================================================


class CVAEPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    Modèle MLflow pyfunc encapsulant le CVAE Fashion-MNIST.

    Contrairement au VAE, chaque requête doit préciser la classe
    Fashion-MNIST demandée (0 à 9).
    """

    FASHION_MNIST_CLASSES = [
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

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Charge le checkpoint .pt du CVAE.
        """

        checkpoint_path = context.artifacts["checkpoint"]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        configuration = checkpoint["configuration"]

        self.model = CVAE(
            latent_dim=configuration["latent_dim"],
            hidden_dim=configuration["hidden_dim"],
            num_classes=configuration.get("num_classes", 10),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.latent_dim = configuration["latent_dim"]

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """
        Génère une image par ligne reçue.

        model_input doit contenir :
        - "class" : entier entre 0 et 9 (obligatoire) ;
        - "z" : vecteur latent optionnel.
        """

        if "class" not in model_input.columns:
            raise ValueError(
                "La colonne 'class' est obligatoire pour le CVAE "
                "(entier entre 0 et 9)."
            )

        if "z" not in model_input.columns:
            model_input = model_input.copy()
            model_input["z"] = None

        results: list[dict] = []

        for _, row in model_input.iterrows():

            requested_class = int(row["class"])

            if not (0 <= requested_class <= 9):
                raise ValueError(
                    "'class' doit être un entier entre 0 et 9. "
                    f"Valeur reçue : {requested_class}."
                )

            z = _parse_optional_latent_vector(
                raw_value=row["z"],
                latent_dim=self.latent_dim,
            )

            if z is None:
                z = torch.randn(1, self.latent_dim)

            labels = torch.tensor([requested_class], dtype=torch.long)

            with torch.no_grad():
                generated_image = self.model.decode(z, labels)

            results.append(
                {
                    "image_base64": _tensor_image_to_base64_png(
                        generated_image[0]
                    ),
                    "image_format": "png",
                    "width": 28,
                    "height": 28,
                    "requested_class": requested_class,
                    "requested_class_name": self.FASHION_MNIST_CLASSES[
                        requested_class
                    ],
                }
            )

        return results