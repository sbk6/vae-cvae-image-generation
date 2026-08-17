"""Encodage des images produites par les modeles.

Toute la logique d'inference vit desormais dans les adaptateurs
(`backend/adapters/`), qui savent chacun quelle plage de valeurs leur modele
produit. Ce module ne s'occupe plus que de la conversion en PNG.

Les images sont renvoyees a leur resolution native (28x28), soit ~200 octets
chacune. C'est le frontend qui les agrandit en CSS avec
`image-rendering: pixelated`, ce qui reste net et evite d'envoyer des images
upscalees inutilement lourdes.
"""
from __future__ import annotations

import base64
import io
from typing import List

import numpy as np
import torch
from PIL import Image

from backend.adapters import ModelAdapter


def _array_to_png_b64(array: np.ndarray) -> str:
    """Encode un tableau (H, W) ou (C, H, W) en uint8 vers un data-URI PNG."""
    if array.ndim == 3:
        if array.shape[0] == 1:
            image = Image.fromarray(array[0], mode="L")
        else:
            image = Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB")
    else:
        image = Image.fromarray(array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def tensor_to_png_b64(image: torch.Tensor, adapter: ModelAdapter) -> str:
    """Convertit une image (C, H, W) produite par `adapter` en PNG base64.

    La conversion passe par `adapter.to_display_range` : c'est ce qui evite
    d'interpreter une sortie Sigmoid [0, 1] comme une sortie Tanh [-1, 1],
    erreur qui produirait une image delavee sans lever d'exception.
    """
    normalized = adapter.to_display_range(image.detach().cpu().float())
    array = (normalized * 255.0).round().to(torch.uint8).numpy()
    return _array_to_png_b64(array)


def batch_to_png_b64(images: torch.Tensor, adapter: ModelAdapter) -> List[str]:
    """Convertit un batch (N, C, H, W) en liste de PNG base64."""
    return [tensor_to_png_b64(images[index], adapter) for index in range(images.size(0))]


def raw_to_png_b64(image_uint8: np.ndarray) -> str:
    """Encode une image brute du fixture, sans passer par un modele.

    Sert a afficher les vignettes d'images reelles : elles sont deja en
    uint8 [0, 255] et ne dependent d'aucune normalisation de modele.
    """
    return _array_to_png_b64(np.asarray(image_uint8, dtype=np.uint8))


def alphas_for(steps: int) -> List[float]:
    """Valeurs de alpha correspondant aux images renvoyees par `interpolate`."""
    return torch.linspace(0.0, 1.0, steps).tolist()
