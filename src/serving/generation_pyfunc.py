"""Wrapper MLflow pyfunc exposant, sous UN SEUL endpoint, les deux
fonctionnalités de démo demandées par l'énoncé, pour chaque dataset de
l'équipe (MNIST aujourd'hui, Fashion-MNIST et CelebA à venir) :

- action="generate"    : CVAE -> choisir une classe, obtenir une image de
  cette classe ("sélection d'une classe cible -> génération d'images").
- action="interpolate"  : VAE -> deux classes + une position t dans [0, 1],
  obtenir l'image intermédiaire dans l'espace latent ("slider
  d'interpolation dans l'espace latent").

Un seul serveur, un seul port, un seul contrat d'API pour l'application web,
qui route en interne vers le bon modèle et la bonne action. Voir
docs/DEPLOIEMENT.md pour le détail complet du contrat.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

import mlflow.pyfunc


class ImageGenerationService(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        # `code_paths=["src"]` (voir scripts/register_generation_model.py) rend
        # le package `src` importable ici, comme dans le reste du dépôt.
        from src.models.cvae import CVAE
        from src.models.vae import VAE

        with open(context.artifacts["registry"], "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)["datasets"]

        self.cvae_models = {}
        self.vae_models = {}
        self.num_classes = {}
        self.latent_centroids = {}
        loaded, skipped = [], []

        for name in registry:
            cvae_ckpt_key, cvae_cfg_key = f"cvae_checkpoint_{name}", f"cvae_config_{name}"
            vae_ckpt_key, vae_cfg_key, centroids_key = (
                f"vae_checkpoint_{name}",
                f"vae_config_{name}",
                f"latent_centroids_{name}",
            )

            if cvae_ckpt_key not in context.artifacts or cvae_cfg_key not in context.artifacts:
                skipped.append(name)
                continue

            with open(context.artifacts[cvae_cfg_key], "r", encoding="utf-8") as f:
                cvae_config = yaml.safe_load(f)
            cvae = CVAE(
                channels=cvae_config["dataset"]["channels"],
                image_size=tuple(cvae_config["dataset"]["image_size"]),
                latent_dim=cvae_config["model"]["latent_dim"],
                num_conditions=cvae_config["dataset"]["num_classes"],
                condition_mode=cvae_config["dataset"].get("condition_mode", "one_hot"),
                hidden_channels=cvae_config["model"]["hidden_channels"],
            )
            cvae_ckpt = torch.load(context.artifacts[cvae_ckpt_key], map_location="cpu")
            cvae.load_state_dict(cvae_ckpt["model_state_dict"])
            cvae.eval()
            self.cvae_models[name] = cvae
            self.num_classes[name] = cvae_config["dataset"]["num_classes"]
            loaded.append(name)

            if vae_ckpt_key in context.artifacts and vae_cfg_key in context.artifacts and centroids_key in context.artifacts:
                with open(context.artifacts[vae_cfg_key], "r", encoding="utf-8") as f:
                    vae_config = yaml.safe_load(f)
                vae = VAE(
                    channels=vae_config["dataset"]["channels"],
                    image_size=tuple(vae_config["dataset"]["image_size"]),
                    latent_dim=vae_config["model"]["latent_dim"],
                    hidden_channels=vae_config["model"]["hidden_channels"],
                )
                vae_ckpt = torch.load(context.artifacts[vae_ckpt_key], map_location="cpu")
                vae.load_state_dict(vae_ckpt["model_state_dict"])
                vae.eval()
                self.vae_models[name] = vae
                with open(context.artifacts[centroids_key], "r", encoding="utf-8") as f:
                    self.latent_centroids[name] = {
                        int(k): torch.tensor(v) for k, v in json.load(f).items()
                    }

        if not self.cvae_models:
            raise RuntimeError(
                "Aucun modèle disponible : vérifier configs/deployment_registry.yaml "
                "et l'existence des fichiers référencés."
            )
        if skipped:
            print(f"[generation_pyfunc] datasets non chargés (fichiers absents) : {skipped}")
        print(f"[generation_pyfunc] datasets disponibles (génération) : {sorted(self.cvae_models.keys())}")
        print(f"[generation_pyfunc] datasets disponibles (interpolation) : {sorted(self.vae_models.keys())}")

    def _tensor_to_base64_png(self, image_tensor: torch.Tensor) -> str:
        pixels = ((image_tensor.numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        if pixels.shape[0] == 1:
            image = Image.fromarray(pixels[0], mode="L")
        else:
            image = Image.fromarray(np.transpose(pixels, (1, 2, 0)), mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _generate(self, dataset: str, classe: int) -> str:
        if dataset not in self.cvae_models:
            available = ", ".join(sorted(self.cvae_models.keys()))
            raise ValueError(f"dataset '{dataset}' indisponible pour la génération. Datasets chargés : {available}")

        num_classes = self.num_classes[dataset]
        if not (0 <= classe < num_classes):
            raise ValueError(f"classe doit être comprise entre 0 et {num_classes - 1} pour '{dataset}', reçu {classe}")

        with torch.no_grad():
            sample = self.cvae_models[dataset].sample(classe, n=1)[0]  # (C, H, W), valeurs dans [-1, 1]
        return self._tensor_to_base64_png(sample)

    def _interpolate(self, dataset: str, classe_a: int, classe_b: int, t: float) -> str:
        if dataset not in self.vae_models:
            available = ", ".join(sorted(self.vae_models.keys())) or "aucun"
            raise ValueError(f"dataset '{dataset}' indisponible pour l'interpolation. Datasets chargés : {available}")

        centroids = self.latent_centroids[dataset]
        num_classes = self.num_classes[dataset]
        for classe in (classe_a, classe_b):
            if not (0 <= classe < num_classes):
                raise ValueError(f"classe doit être comprise entre 0 et {num_classes - 1} pour '{dataset}', reçu {classe}")
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"t doit être compris entre 0 et 1, reçu {t}")

        z = (1 - t) * centroids[classe_a] + t * centroids[classe_b]
        with torch.no_grad():
            image = self.vae_models[dataset].decode(z.unsqueeze(0))[0]  # (C, H, W)
        return self._tensor_to_base64_png(image)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        results = []
        for record in model_input.to_dict("records"):
            action = record.get("action") or "generate"
            dataset = record.get("dataset") or "mnist"

            if action == "generate":
                classe = record.get("classe")
                if pd.isna(classe):
                    raise ValueError("action='generate' nécessite le champ 'classe'")
                results.append(self._generate(dataset, int(classe)))
            elif action == "interpolate":
                classe_a, classe_b, t = record.get("classe_a"), record.get("classe_b"), record.get("t")
                if pd.isna(classe_a) or pd.isna(classe_b) or pd.isna(t):
                    raise ValueError("action='interpolate' nécessite les champs 'classe_a', 'classe_b' et 't'")
                results.append(self._interpolate(dataset, int(classe_a), int(classe_b), float(t)))
            else:
                raise ValueError(f"action '{action}' inconnue. Valeurs possibles : 'generate', 'interpolate'")

        return pd.Series(results, name="image_base64")


def build_artifacts(registry_path: str) -> dict:
    """Construit le dict d'artefacts à passer à `mlflow.pyfunc.log_model`,
    en n'incluant que les fichiers qui existent réellement sur disque.
    """
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)["datasets"]

    artifacts = {"registry": registry_path}
    generate_ok, interpolate_ok, missing = [], [], []

    for name, entry in registry.items():
        cvae_entry = entry.get("cvae", {})
        cvae_ckpt, cvae_cfg = Path(cvae_entry.get("checkpoint", "")), Path(cvae_entry.get("config", ""))
        if cvae_entry and cvae_ckpt.exists() and cvae_cfg.exists():
            artifacts[f"cvae_checkpoint_{name}"] = str(cvae_ckpt)
            artifacts[f"cvae_config_{name}"] = str(cvae_cfg)
            generate_ok.append(name)
        else:
            missing.append(f"{name} (generate)")
            continue  # sans CVAE, ce dataset est ignoré entièrement (voir load_context)

        vae_entry = entry.get("vae", {})
        vae_ckpt = Path(vae_entry.get("checkpoint", ""))
        vae_cfg = Path(vae_entry.get("config", ""))
        centroids = Path(vae_entry.get("latent_centroids", ""))
        if vae_entry and vae_ckpt.exists() and vae_cfg.exists() and centroids.exists():
            artifacts[f"vae_checkpoint_{name}"] = str(vae_ckpt)
            artifacts[f"vae_config_{name}"] = str(vae_cfg)
            artifacts[f"latent_centroids_{name}"] = str(centroids)
            interpolate_ok.append(name)
        else:
            missing.append(f"{name} (interpolate)")

    if not generate_ok:
        raise RuntimeError("Aucun dataset avec un CVAE disponible dans le registre.")
    print(f"Datasets inclus pour la génération (CVAE) : {generate_ok}")
    print(f"Datasets inclus pour l'interpolation (VAE) : {interpolate_ok}")
    if missing:
        print(f"Non inclus (fichiers manquants) : {missing}")
    return artifacts
