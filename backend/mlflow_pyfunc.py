"""Enveloppe MLflow generique au-dessus de `ModelAdapter`.

Le depot contient trois familles de modeles aux interfaces incompatibles
(MNIST, Fashion-MNIST, CelebA), deja reconciliees par `backend/adapters/`.
Ce module fait le pas suivant : exposer **n'importe quel** adaptateur comme un
`mlflow.pyfunc.PythonModel`, pour que l'application n'accede plus jamais a un
modele autrement que par le Model Registry.

Pourquoi une enveloppe generique plutot qu'une par famille : les wrappers de
`projects/david_fashion_mnist/deployment/` sont specifiques a Fashion-MNIST et
n'exposent que `decode`. Reconstruction et interpolation depuis de vraies
images exigent `encode`. Plutot que de modifier le code d'un coequipier, on
enveloppe la couche d'abstraction qui existe deja et qui couvre les trois
familles d'un coup.

Le contrat d'entree/sortie est volontairement serialisable en JSON, pour que
le modele reste servable tel quel par `mlflow models serve` :

    colonnes d'entree : op, z, class_label, image_base64, n, seed
    sortie            : liste de dicts {images_base64, z, ...}
"""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image

OP_DECODE = "decode"
OP_ENCODE = "encode"
OP_SAMPLE = "sample"
SUPPORTED_OPS = (OP_DECODE, OP_ENCODE, OP_SAMPLE)

MAX_BATCH = 64


def image_to_base64_png(array_uint8: np.ndarray) -> str:
    """Encode un tableau (H, W) ou (H, W, 3) uint8 en PNG base64, sans prefixe data:."""
    if array_uint8.ndim == 2:
        image = Image.fromarray(array_uint8, mode="L")
    elif array_uint8.ndim == 3 and array_uint8.shape[2] == 1:
        image = Image.fromarray(array_uint8[:, :, 0], mode="L")
    else:
        image = Image.fromarray(array_uint8, mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_png_to_array(encoded: str) -> np.ndarray:
    """Decode un PNG base64 vers un tableau uint8 (H, W) ou (H, W, 3)."""
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    return np.array(Image.open(io.BytesIO(raw)))


def _coerce_optional(value: Any) -> Any:
    """pandas represente une valeur absente par None ou NaN selon le chemin d'entree."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and value == "":
        return None
    return value


def parse_latent(value: Any, latent_dim: int) -> Optional[torch.Tensor]:
    """Valide un vecteur latent recu en JSON, ou renvoie None s'il est absent.

    MLflow ne valide pas le contenu des colonnes : sans ce controle, un
    payload malforme remonterait en erreur interne depuis torch au lieu d'un
    message lisible.
    """
    value = _coerce_optional(value)
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("'z' doit etre une liste de nombres.")
    if len(value) != latent_dim:
        raise ValueError(f"'z' doit contenir exactement {latent_dim} valeurs (recu {len(value)}).")

    cleaned: List[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float, np.floating, np.integer)):
            raise ValueError(f"'z[{index}]' doit etre un nombre.")
        item = float(item)
        if not np.isfinite(item):
            raise ValueError(f"'z[{index}]' doit etre fini (NaN et inf interdits).")
        cleaned.append(item)
    return torch.tensor(cleaned, dtype=torch.float32)


class AdapterPyfuncModel(mlflow.pyfunc.PythonModel):
    """Expose un `ModelAdapter` comme modele MLflow.

    L'adaptateur est reconstruit au chargement a partir de deux artefacts :
    - `checkpoint` : les poids ;
    - `spec` : un JSON indiquant quel loader de `backend.adapters.LOADERS`
      employer. Un artefact `config` optionnel porte le YAML, necessaire pour
      la seule famille MNIST, les deux autres embarquant leur configuration
      dans le checkpoint.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        # Import differe : `code_paths` n'est ajoute au sys.path qu'au moment
        # ou MLflow charge le modele, pas a l'import de ce module.
        from backend.adapters import LOADERS

        with open(context.artifacts["spec"], "r", encoding="utf-8") as handle:
            spec = json.load(handle)

        self.spec = spec
        self.model_id = spec["model_id"]
        self.dataset_id = spec["dataset_id"]

        loader = LOADERS[spec["loader"]]
        self.adapter = loader(
            context.artifacts["checkpoint"],
            torch.device("cpu"),
            config_path=context.artifacts.get("config"),
        )

    # ------------------------------------------------------------ helpers

    def _decode_to_base64(self, z: torch.Tensor, class_label: Optional[int]) -> List[str]:
        images = self.adapter.decode(z, class_label)
        displayed = self.adapter.to_display_range(images.detach().cpu().float())
        array = (displayed * 255.0).round().to(torch.uint8).numpy()

        encoded: List[str] = []
        for index in range(array.shape[0]):
            single = array[index]
            # (C, H, W) -> (H, W) en niveaux de gris, (H, W, C) en couleur.
            single = single[0] if single.shape[0] == 1 else np.transpose(single, (1, 2, 0))
            encoded.append(image_to_base64_png(single))
        return encoded

    def _row_class_label(self, row: pd.Series) -> Optional[int]:
        value = _coerce_optional(row.get("class_label"))
        if value is None:
            if self.adapter.is_conditional:
                raise ValueError("Ce modele est conditionnel : 'class_label' est obligatoire.")
            return None
        if not self.adapter.is_conditional:
            return None
        label = int(value)
        if not 0 <= label < self.adapter.num_conditions:
            raise ValueError(
                f"'class_label' doit etre compris entre 0 et "
                f"{self.adapter.num_conditions - 1} (recu {label})."
            )
        return label

    # ------------------------------------------------------------ predict

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """Une ligne d'entree produit une entree de sortie."""
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        results: List[Dict[str, Any]] = []
        for _, row in model_input.iterrows():
            op = _coerce_optional(row.get("op")) or OP_SAMPLE
            if op not in SUPPORTED_OPS:
                raise ValueError(
                    f"Operation inconnue : {op}. Attendu : {', '.join(SUPPORTED_OPS)}."
                )
            if op == OP_ENCODE:
                results.append(self._handle_encode(row))
            else:
                results.append(self._handle_decode(row))
        return results

    def _handle_encode(self, row: pd.Series) -> Dict[str, Any]:
        encoded_image = _coerce_optional(row.get("image_base64"))
        if encoded_image is None:
            raise ValueError("L'operation 'encode' exige 'image_base64'.")

        array = base64_png_to_array(str(encoded_image))
        tensor = self.adapter.prepare_input(array)
        mu = self.adapter.encode(tensor, self._row_class_label(row))
        return {
            "z": mu[0].detach().cpu().tolist(),
            "model_id": self.model_id,
            "latent_dim": self.adapter.latent_dim,
        }

    def _handle_decode(self, row: pd.Series) -> Dict[str, Any]:
        latent_dim = self.adapter.latent_dim
        class_label = self._row_class_label(row)
        z = parse_latent(row.get("z"), latent_dim)

        if z is None:
            # Aucun z fourni : on tire dans la prior. La seed rend le tirage
            # reproductible sans toucher au RNG global du processus.
            count = _coerce_optional(row.get("n"))
            count = 1 if count is None else int(count)
            if not 1 <= count <= MAX_BATCH:
                raise ValueError(f"'n' doit etre compris entre 1 et {MAX_BATCH} (recu {count}).")
            seed = _coerce_optional(row.get("seed"))
            batch = self.adapter.sample_latent(count, None if seed is None else int(seed))
        else:
            batch = z.unsqueeze(0)

        images = self._decode_to_base64(batch, class_label)
        return {
            "images_base64": images,
            "image_base64": images[0],
            "image_format": "png",
            "model_id": self.model_id,
            "class_label": class_label,
            "latent_dim": latent_dim,
        }
