"""API FastAPI de demonstration des modeles VAE / CVAE de l'equipe.

Trois datasets sont servis par la meme API : MNIST (`src/`), Fashion-MNIST
(`projects/david_fashion_mnist/`) et CelebA (`projects/blaise_celeba/`).

**Toute l'inference passe par MLflow.** Aucune route ci-dessous n'importe ni
n'instancie un modele : chacune obtient un `mlflow.pyfunc` charge depuis le
Model Registry via `backend/mlflow_registry.py`. Le catalogue ne sert plus
qu'a decrire ce qui existe (libelles, datasets, series d'ablation) et a savoir
quoi enregistrer.

Le contrat du modele packagé est defini dans `backend/mlflow_pyfunc.py` : une
colonne `op` (sample, decode, encode) et des colonnes facultatives. Il est
identique en appel direct et en HTTP, donc les memes modeles restent servables
par `mlflow models serve`.

Deux modes de service :
- developpement : le frontend tourne sur Vite (:5173) et appelle cette API
  (:8000) en cross-origin, d'ou CORSMiddleware ;
- demonstration : le build React est servi par l'API, tout tient sur un port.
"""
from __future__ import annotations

import csv
import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.catalog import Catalog, ModelEntry
from backend.fixtures import FixtureRegistry
from backend.mlflow_pyfunc import base64_png_to_array, image_to_base64_png
from backend.mlflow_registry import RegistryGateway, is_registry_available, tracking_uri

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
FASHION_RESULTS = ROOT_DIR / "projects" / "david_fashion_mnist" / "results"
CELEBA_RESULTS = ROOT_DIR / "projects" / "blaise_celeba" / "results"

MAX_BATCH = 64
MAX_STEPS = 32

DATA_URI_PREFIX = "data:image/png;base64,"


def as_data_uri(encoded: str) -> str:
    """Le modele MLflow renvoie du base64 nu ; le navigateur attend un data-URI."""
    return DATA_URI_PREFIX + encoded


def normalize_ablation(rows: List[dict]) -> List[dict]:
    """Ramene les tableaux d'ablation a une seule forme.

    Les sous-projets ne journalisent pas la meme chose : MNIST retient la
    derniere epoch (`final_val_*`), CelebA la meilleure (`best_val_*`). Plutot
    que d'apprendre les deux conventions au frontend, on les aplatit ici.
    """
    normalized = []
    for row in rows:
        normalized.append(
            {
                "beta": row.get("beta"),
                "val_loss": row.get("final_val_loss", row.get("best_val_loss")),
                "val_reconstruction": row.get(
                    "final_val_reconstruction", row.get("best_val_reconstruction")
                ),
                "val_kl": row.get("final_val_kl", row.get("best_val_kl")),
                "best_epoch": row.get("best_epoch"),
            }
        )
    return normalized


# --------------------------------------------------------------------- #
# Schemas de requete
# --------------------------------------------------------------------- #

class SampleRequest(BaseModel):
    model_id: str
    n: int = Field(default=8, ge=1, le=MAX_BATCH)
    class_label: Optional[int] = Field(default=None, ge=0)
    seed: Optional[int] = None


class DecodeRequest(BaseModel):
    model_id: str
    z: List[float]
    class_label: Optional[int] = Field(default=None, ge=0)


class FixtureRequest(BaseModel):
    model_id: str
    index: int = Field(ge=0)
    class_label: Optional[int] = Field(default=None, ge=0)


class InterpolateRequest(BaseModel):
    model_id: str
    source_index: int = Field(ge=0)
    target_index: int = Field(ge=0)
    steps: int = Field(default=12, ge=2, le=MAX_STEPS)
    class_label: Optional[int] = Field(default=None, ge=0)


class AblationRequest(BaseModel):
    dataset: str = "mnist"
    series: Optional[str] = None
    z: Optional[List[float]] = None
    seed: Optional[int] = None
    class_label: Optional[int] = Field(default=None, ge=0)


# --------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------- #

def create_app(device: Optional[str] = None, warmup: bool = True) -> FastAPI:
    catalog = Catalog(device=device)
    fixtures = FixtureRegistry()
    gateway = RegistryGateway()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Le tout premier chargement MLflow coute une trentaine de secondes
        # (initialisation du store et de ses dependances), les suivants moins
        # d'une seconde. On l'absorbe au demarrage plutot que sur le premier
        # clic de l'utilisateur.
        #
        # Dans un thread separe, et non directement ici : le lifespan bloque
        # l'acceptation des connexions tant qu'il n'a pas rendu la main, et le
        # serveur paraitrait mort pendant tout le prechauffage.
        def preload() -> None:
            try:
                models = catalog.models()
                if models:
                    gateway.load(models[0].model_id)
            except Exception as error:  # pragma: no cover - demarrage degrade
                print(f"Prechauffage MLflow ignore : {error}")

        if warmup and is_registry_available():
            threading.Thread(target=preload, name="mlflow-warmup", daemon=True).start()
        yield

    app = FastAPI(
        title="Demonstration VAE / CVAE",
        description=(
            "Generation d'images a partir des modeles VAE et CVAE entraines par l'equipe "
            "sur MNIST, Fashion-MNIST et CelebA. Toute l'inference passe par le MLflow "
            "Model Registry."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------- #
    # Aides communes
    # ----------------------------------------------------------------- #

    def resolve_entry(model_id: str) -> ModelEntry:
        try:
            return catalog.entry(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Modele inconnu : {model_id}")

    def resolve_model(model_id: str):
        """Renvoie (entree de catalogue, modele pyfunc, metadonnees du Registry)."""
        entry = resolve_entry(model_id)
        try:
            return entry, gateway.load(model_id), gateway.describe(model_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
        except KeyError as error:
            raise HTTPException(status_code=503, detail=str(error).strip("'"))

    def resolve_fixtures(dataset_id: str):
        try:
            dataset = catalog.dataset(dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error))
        store = fixtures.get(dataset_id, dataset.fixture_name)
        if not store.available:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Fixture d'images absent pour {dataset.label}. Le generer avec : "
                    "python scripts/build_demo_fixtures.py"
                ),
            )
        return store

    def check_class_label(description: Dict[str, Any], class_label: Optional[int]) -> Optional[float]:
        """Valide la classe demandee au regard des metadonnees du Registry."""
        if not description.get("conditional"):
            return None
        if class_label is None:
            raise HTTPException(
                status_code=400,
                detail="Ce modele est conditionnel : 'class_label' est obligatoire.",
            )
        maximum = int(description.get("num_conditions") or 0) - 1
        if not 0 <= class_label <= maximum:
            raise HTTPException(
                status_code=400,
                detail=f"'class_label' doit etre compris entre 0 et {maximum} (recu {class_label}).",
            )
        return float(class_label)

    def check_latent(description: Dict[str, Any], z: List[float]) -> str:
        """Verifie la dimension du vecteur latent et le serialise pour MLflow."""
        expected = int(description.get("latent_dim") or 0)
        if expected and len(z) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"'z' doit contenir exactement {expected} valeurs (recu {len(z)}).",
            )
        return json.dumps(z)

    def invoke(model, **columns) -> Dict[str, Any]:
        """Appelle le modele MLflow avec une ligne complete.

        Toutes les colonnes de la signature doivent etre presentes, meme
        vides : MLflow refuse une entree dont une colonne declaree manque.
        """
        row = {
            "op": "sample",
            "z": None,
            "class_label": None,
            "image_base64": None,
            "n": None,
            "seed": None,
        }
        row.update(columns)
        try:
            return model.predict(pd.DataFrame([row]))[0]
        except HTTPException:
            raise
        except Exception as error:
            # Les erreurs metier levees dans le modele (classe absente, z
            # malforme) sont des erreurs de requete, pas des pannes serveur.
            raise HTTPException(status_code=400, detail=str(error).split("\n")[-1][:300])

    def encode_fixture_image(model, store, index: int, class_label: Optional[float]) -> List[float]:
        """Encode une image du fixture en passant par le modele MLflow."""
        try:
            raw = store.raw(index)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        payload = invoke(
            model,
            op="encode",
            image_base64=image_to_base64_png(raw),
            class_label=class_label,
        )
        return payload["z"]

    # ----------------------------------------------------------------- #
    # Metadonnees
    # ----------------------------------------------------------------- #

    @app.get("/api/health", tags=["metadonnees"])
    def health():
        return {
            "status": "ok",
            "device": str(catalog.device),
            "torch": torch.__version__,
            "inference": "mlflow-registry",
            "registry_available": is_registry_available(),
            "registry_uri": tracking_uri(),
            "registered_models": gateway.available_names(),
            "loaded_models": gateway.loaded_ids(),
            "datasets": [dataset.dataset_id for dataset in catalog.available_datasets()],
        }

    @app.get("/api/datasets", tags=["metadonnees"])
    def list_datasets():
        """Datasets a proposer dans l'interface.

        Un dataset est annonce des qu'il a quelque chose a montrer : des modeles
        servables, ou a defaut des images reelles et des resultats d'evaluation.
        Le masquer tant qu'aucun poids n'est arrive rendrait invisible le reste
        du travail de son auteur — metriques, ablation, figures — alors que
        l'API le sert deja. Le champ `model_count` dit a l'interface quels
        ecrans elle peut reellement proposer.
        """
        payload = []
        for dataset in catalog.datasets.values():
            store = fixtures.get(dataset.dataset_id, dataset.fixture_name)
            metadata = catalog.dataset_metadata(dataset)
            metadata["fixtures_available"] = store.available
            if metadata["model_count"] == 0 and not store.available:
                continue
            payload.append(metadata)

        if not payload:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Aucun dataset exploitable. Generer les fixtures avec "
                    "'python scripts/build_demo_fixtures.py', puis enregistrer les modeles "
                    "avec 'python scripts/register_models.py'."
                ),
            )
        return {"datasets": payload}

    @app.get("/api/models", tags=["metadonnees"])
    def list_models(dataset: Optional[str] = Query(default=None)):
        try:
            models = catalog.models(dataset)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error))
        # Liste vide plutot qu'une erreur : le dataset existe, ses poids ne sont
        # simplement pas encore la. L'interface le signale et affiche ce qui
        # reste consultable (metriques, images reelles).
        payload = []
        for entry in models:
            metadata = catalog.model_metadata(entry)
            # Les dimensions viennent du Registry, pas d'un adaptateur charge.
            try:
                description = gateway.describe(entry.model_id)
                metadata.update(
                    {
                        "latent_dim": description["latent_dim"],
                        "num_conditions": description["num_conditions"],
                        "registered_name": description["registered_name"],
                        "version": description["version"],
                        "registered": True,
                    }
                )
            except Exception:
                # Checkpoint present mais pas encore empaquete : on l'annonce
                # plutot que de le masquer, l'action corrective etant connue.
                metadata["registered"] = False
            payload.append(metadata)
        return {"models": payload}

    @app.get("/api/metrics", tags=["metadonnees"])
    def metrics(dataset: str = Query(default="mnist")):
        result: Dict[str, Any] = {}

        if dataset == "mnist":
            comparison = ROOT_DIR / "reports" / "experiments" / "comparison.json"
            if comparison.exists():
                result["comparison"] = json.loads(comparison.read_text(encoding="utf-8"))
            ablation = ROOT_DIR / "reports" / "experiments" / "ablation" / "results.json"
            if ablation.exists():
                result["ablation"] = normalize_ablation(
                    json.loads(ablation.read_text(encoding="utf-8"))
                )

        elif dataset == "fashion_mnist":
            # David produit ses resultats en CSV : on les convertit ici pour que
            # le frontend n'ait qu'un seul format a lire.
            evaluation = FASHION_RESULTS / "evaluation_metrics.csv"
            if evaluation.exists():
                with open(evaluation, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                result["evaluation"] = [
                    {
                        "checkpoint": row["checkpoint"],
                        "model_type": row["model_type"],
                        "beta": float(row["beta"]),
                        "test_reconstruction": float(row["test_reconstruction"]),
                        "test_kl": float(row["test_kl"]),
                        "test_total": float(row["test_total"]),
                        "n_test": int(row["processed_test_images"]),
                    }
                    for row in rows
                ]

        elif dataset == "celeba":
            comparison = CELEBA_RESULTS / "experiments" / "comparison.json"
            if comparison.exists():
                result["comparison"] = json.loads(comparison.read_text(encoding="utf-8"))
            ablation = CELEBA_RESULTS / "experiments" / "ablation" / "results.json"
            if ablation.exists():
                result["ablation"] = normalize_ablation(
                    json.loads(ablation.read_text(encoding="utf-8"))
                )
        else:
            raise HTTPException(status_code=404, detail=f"Dataset inconnu : {dataset}")

        if not result:
            raise HTTPException(
                status_code=503,
                detail="Aucun resultat d'evaluation disponible pour ce dataset.",
            )
        return result

    @app.get("/api/fixtures", tags=["metadonnees"])
    def list_fixtures(dataset: str = Query(default="mnist")):
        store = resolve_fixtures(dataset)
        dataset_entry = catalog.dataset(dataset)
        grouped = store.indices_by_class()
        payload = {
            str(label): [
                {"index": index, "image": as_data_uri(image_to_base64_png(store.raw(index)))}
                for index in indices
            ]
            for label, indices in sorted(grouped.items())
        }
        return {
            "by_class": payload,
            "count": len(store),
            "class_names": dataset_entry.class_names,
        }

    # ----------------------------------------------------------------- #
    # Generation
    # ----------------------------------------------------------------- #

    @app.post("/api/sample", tags=["generation"])
    def sample(request: SampleRequest):
        entry, model, description = resolve_model(request.model_id)
        class_label = check_class_label(description, request.class_label)

        payload = invoke(
            model,
            op="sample",
            n=float(request.n),
            seed=None if request.seed is None else float(request.seed),
            class_label=class_label,
        )
        return {
            "images": [as_data_uri(image) for image in payload["images_base64"]],
            "model_id": entry.model_id,
            "class_label": request.class_label if description.get("conditional") else None,
            "seed": request.seed,
        }

    @app.post("/api/decode", tags=["generation"])
    def decode(request: DecodeRequest):
        entry, model, description = resolve_model(request.model_id)
        class_label = check_class_label(description, request.class_label)
        payload = invoke(
            model,
            op="decode",
            z=check_latent(description, request.z),
            class_label=class_label,
        )
        return {"image": as_data_uri(payload["image_base64"]), "model_id": entry.model_id}

    @app.post("/api/encode", tags=["generation"])
    def encode(request: FixtureRequest):
        entry, model, description = resolve_model(request.model_id)
        store = resolve_fixtures(entry.dataset_id)

        class_label = request.class_label
        if description.get("conditional") and class_label is None:
            class_label = store.label_of(request.index)
        checked = check_class_label(description, class_label)

        z = encode_fixture_image(model, store, request.index, checked)
        return {
            "z": z,
            "index": request.index,
            "true_label": store.label_of(request.index),
            "model_id": entry.model_id,
        }

    @app.post("/api/reconstruct", tags=["generation"])
    def reconstruct(request: FixtureRequest):
        entry, model, description = resolve_model(request.model_id)
        store = resolve_fixtures(entry.dataset_id)

        class_label = request.class_label
        if description.get("conditional") and class_label is None:
            class_label = store.label_of(request.index)
        checked = check_class_label(description, class_label)

        z = encode_fixture_image(model, store, request.index, checked)
        payload = invoke(model, op="decode", z=json.dumps(z), class_label=checked)

        return {
            # L'original vient du fixture brut : il ne depend d'aucun modele.
            "original": as_data_uri(image_to_base64_png(store.raw(request.index))),
            "reconstruction": as_data_uri(payload["image_base64"]),
            "index": request.index,
            "true_label": store.label_of(request.index),
            "model_id": entry.model_id,
        }

    @app.post("/api/interpolate", tags=["generation"])
    def interpolate(request: InterpolateRequest):
        entry, model, description = resolve_model(request.model_id)
        store = resolve_fixtures(entry.dataset_id)

        conditional = bool(description.get("conditional"))
        class_label = request.class_label
        if conditional and class_label is None:
            # Condition figee sur la classe de depart : on observe alors la
            # morphologie encodee dans z, pas le changement de condition.
            class_label = store.label_of(request.source_index)
        checked = check_class_label(description, class_label)

        source_label = float(store.label_of(request.source_index)) if conditional else None
        target_label = float(store.label_of(request.target_index)) if conditional else None
        z_source = encode_fixture_image(model, store, request.source_index, source_label)
        z_target = encode_fixture_image(model, store, request.target_index, target_label)

        # L'interpolation est calculee ici : le modele MLflow expose decode et
        # encode, la combinaison lineaire de deux latents releve de l'appelant.
        steps = request.steps
        images: List[str] = []
        alphas: List[float] = []
        for index in range(steps):
            alpha = index / (steps - 1)
            alphas.append(alpha)
            z = [(1.0 - alpha) * a + alpha * b for a, b in zip(z_source, z_target)]
            payload = invoke(model, op="decode", z=json.dumps(z), class_label=checked)
            images.append(as_data_uri(payload["image_base64"]))

        return {
            "images": images,
            "alphas": alphas,
            "source": {
                "index": request.source_index,
                "label": store.label_of(request.source_index),
                "image": as_data_uri(image_to_base64_png(store.raw(request.source_index))),
            },
            "target": {
                "index": request.target_index,
                "label": store.label_of(request.target_index),
                "image": as_data_uri(image_to_base64_png(store.raw(request.target_index))),
            },
            "model_id": entry.model_id,
        }

    @app.post("/api/ablation/compare", tags=["generation"])
    def ablation_compare(request: AblationRequest):
        try:
            candidates = [model for model in catalog.models(request.dataset) if model.ablation_series]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error))

        # Les modeles sont regroupes par serie : comparer un VAE et un CVAE a
        # beta different melangerait deux effets distincts.
        series: Dict[str, List[ModelEntry]] = {}
        for model in candidates:
            series.setdefault(model.ablation_series, []).append(model)

        if request.series is not None:
            if request.series not in series:
                raise HTTPException(
                    status_code=404,
                    detail=f"Serie d'ablation inconnue : {request.series}. Disponibles : {sorted(series)}.",
                )
            selected = series[request.series]
        else:
            selected = max(series.values(), key=len, default=[])

        if len(selected) < 2:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Serie d'ablation incomplete pour '{request.dataset}' : "
                    f"{len(selected)} modele(s), il en faut au moins 2."
                ),
            )

        selected = sorted(selected, key=lambda model: (model.beta is None, model.beta))
        _, reference_model, reference_description = resolve_model(selected[0].model_id)
        checked = check_class_label(reference_description, request.class_label)

        if request.z is not None:
            z_json = check_latent(reference_description, request.z)
            z_values = request.z
        else:
            # Un z unique est tire une fois puis envoye tel quel a chaque
            # modele : c'est ce qui rend la comparaison entre betas honnete.
            # Le tirage dans la prior N(0, I) n'est pas de l'inference, seule
            # sa dimension depend du modele et elle vient du Registry.
            latent_dim = int(reference_description.get("latent_dim") or 16)
            generator = torch.Generator()
            if request.seed is not None:
                generator.manual_seed(int(request.seed))
            else:
                generator.seed()
            z_values = torch.randn(latent_dim, generator=generator).tolist()
            z_json = json.dumps(z_values)

        results = []
        for entry in selected:
            _, model, description = resolve_model(entry.model_id)
            if description.get("latent_dim") != reference_description.get("latent_dim"):
                # Comparer des latents de dimensions differentes n'aurait aucun sens.
                continue
            label = check_class_label(description, request.class_label)
            payload = invoke(model, op="decode", z=z_json, class_label=label)
            results.append(
                {
                    "model_id": entry.model_id,
                    "label": entry.label,
                    "beta": entry.beta,
                    "description": entry.description,
                    "image": as_data_uri(payload["image_base64"]),
                }
            )

        return {
            "results": results,
            "z": z_values,
            "seed": request.seed,
            "dataset": request.dataset,
            "series": selected[0].ablation_series,
            "available_series": sorted(series),
        }

    # ----------------------------------------------------------------- #
    # Frontend buildé
    # ----------------------------------------------------------------- #

    @app.get("/{requested_path:path}", include_in_schema=False)
    def serve_frontend(requested_path: str = ""):
        if requested_path.startswith("api/"):
            return JSONResponse({"detail": "Endpoint inconnu."}, status_code=404)

        if not FRONTEND_DIST.exists():
            return JSONResponse(
                {
                    "message": "API en ligne. Le frontend n'est pas buildé.",
                    "developpement": "npm run dev dans frontend/, puis http://localhost:5173",
                    "demonstration": "make demo builde le frontend et le sert ici.",
                    "documentation": "/docs",
                    "api": "/api/health",
                }
            )

        candidate = FRONTEND_DIST / requested_path
        if requested_path and candidate.is_file():
            return FileResponse(candidate)
        # Fallback SPA : toute autre route est geree cote client.
        return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="API de demonstration VAE / CVAE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="cpu, cuda ou auto")
    parser.add_argument("--reload", action="store_true", help="Rechargement a chaud (developpement)")
    args = parser.parse_args()

    if args.reload:
        uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=True)
    else:
        print(f"API de demonstration sur http://{args.host}:{args.port}")
        print(f"Documentation interactive sur http://{args.host}:{args.port}/docs")
        uvicorn.run(create_app(device=args.device), host=args.host, port=args.port)
