"""API Flask de demonstration pour les modeles VAE / CVAE de l'equipe.

Deux datasets sont servis par la meme API :
- MNIST, via les modeles de `src/` (sortie Tanh, [-1, 1]) ;
- Fashion-MNIST, via ceux de `projects/david_fashion_mnist/` (Sigmoid, [0, 1]).

Les deux familles ont des interfaces incompatibles ; elles sont uniformisees
par les adaptateurs de `backend/adapters/`. Aucune route ci-dessous ne
manipule un modele directement.

Deux modes de service :
- developpement : le frontend tourne sur Vite (:5173) et appelle cette API
  (:8000) en cross-origin, d'ou flask-cors ;
- demonstration : le build React est servi directement par Flask, tout tient
  sur un seul port et un seul container.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend import inference
from backend.catalog import Catalog
from backend.fixtures import FixtureRegistry

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
FASHION_RESULTS = ROOT_DIR / "projects" / "david_fashion_mnist" / "results"
MAX_BATCH = 64
MAX_STEPS = 32


class ApiError(Exception):
    """Erreur applicative renvoyee en JSON avec un code HTTP explicite."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _payload() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        raise ApiError("Corps de requete JSON attendu.")
    if not isinstance(data, dict):
        raise ApiError("Le corps JSON doit etre un objet.")
    return data


def _require_int(data: Dict[str, Any], key: str, minimum: int, maximum: int, default: Optional[int] = None) -> int:
    if key not in data or data[key] is None:
        if default is None:
            raise ApiError(f"Champ '{key}' manquant.")
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(f"'{key}' doit etre un entier.")
    if value < minimum or value > maximum:
        raise ApiError(f"'{key}' doit etre compris entre {minimum} et {maximum} (recu {value}).")
    return value


def _optional_seed(data: Dict[str, Any]) -> Optional[int]:
    value = data.get("seed")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError("'seed' doit etre un entier ou null.")
    return value


def create_app(device: Optional[str] = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    catalog = Catalog(device=device)
    fixtures = FixtureRegistry()

    # ------------------------------------------------------------------ #
    # Resolution des parametres communs
    # ------------------------------------------------------------------ #

    def resolve_model(data: Dict[str, Any]):
        """Renvoie (entree du catalogue, adaptateur charge)."""
        model_id = data.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ApiError("Champ 'model_id' manquant.")
        try:
            return catalog.entry(model_id), catalog.adapter(model_id)
        except KeyError:
            raise ApiError(f"Modele inconnu : {model_id}", status=404)
        except FileNotFoundError as exc:
            raise ApiError(str(exc), status=503)

    def resolve_fixtures(dataset_id: str):
        dataset = catalog.dataset(dataset_id)
        store = fixtures.get(dataset_id, dataset.fixture_name)
        if not store.available:
            raise ApiError(
                f"Fixture d'images absent pour {dataset.label}. "
                "Le generer avec : python scripts/build_demo_fixtures.py",
                status=503,
            )
        return store

    def resolve_class(data: Dict[str, Any], adapter, required: bool) -> Optional[int]:
        """Valide la classe demandee, pertinente uniquement si le modele est conditionnel."""
        if not adapter.is_conditional:
            return None
        maximum = adapter.num_conditions - 1
        if not required and data.get("class_label") is None:
            return None
        return _require_int(data, "class_label", 0, maximum)

    def encode_fixture(adapter, store, index: int) -> torch.Tensor:
        """Charge une image du fixture et la normalise pour l'adaptateur cible."""
        try:
            raw = store.raw(index)
        except ValueError as exc:
            raise ApiError(str(exc))
        return adapter.prepare_input(raw)

    # ------------------------------------------------------------------ #
    # Metadonnees
    # ------------------------------------------------------------------ #

    @app.get("/api/health")
    def health():
        datasets = catalog.available_datasets()
        return jsonify(
            {
                "status": "ok",
                "device": str(catalog.device),
                "torch": torch.__version__,
                "datasets": [dataset.dataset_id for dataset in datasets],
                "models_available": [model.model_id for model in catalog.models()],
            }
        )

    @app.get("/api/datasets")
    def list_datasets():
        """Datasets disposant d'au moins un checkpoint utilisable."""
        datasets = catalog.available_datasets()
        if not datasets:
            raise ApiError(
                "Aucun checkpoint trouve. Lancer 'make train-vae' et 'make train-cvae', "
                "et deposer les checkpoints Fashion-MNIST dans "
                "projects/david_fashion_mnist/checkpoints/.",
                status=503,
            )
        payload = []
        for dataset in datasets:
            metadata = catalog.dataset_metadata(dataset)
            store = fixtures.get(dataset.dataset_id, dataset.fixture_name)
            metadata["fixtures_available"] = store.available
            payload.append(metadata)
        return jsonify({"datasets": payload})

    @app.get("/api/models")
    def list_models():
        dataset_id = request.args.get("dataset")
        try:
            models = catalog.models(dataset_id)
        except KeyError as exc:
            raise ApiError(str(exc), status=404)
        if not models:
            raise ApiError("Aucun modele disponible pour ce dataset.", status=503)
        return jsonify({"models": [catalog.model_metadata(model) for model in models]})

    @app.get("/api/metrics")
    def metrics():
        """Resultats chiffres deja produits par les scripts d'evaluation."""
        dataset_id = request.args.get("dataset", "mnist")
        result: Dict[str, Any] = {}

        if dataset_id == "mnist":
            comparison = ROOT_DIR / "reports" / "experiments" / "comparison.json"
            if comparison.exists():
                result["comparison"] = json.loads(comparison.read_text(encoding="utf-8"))

            ablation = ROOT_DIR / "reports" / "experiments" / "ablation" / "results.json"
            if ablation.exists():
                result["ablation"] = json.loads(ablation.read_text(encoding="utf-8"))

        elif dataset_id == "fashion_mnist":
            # David produit ses resultats en CSV plutot qu'en JSON : on les
            # convertit ici pour que le frontend n'ait qu'un seul format a lire.
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
        else:
            raise ApiError(f"Dataset inconnu : {dataset_id}", status=404)

        if not result:
            raise ApiError("Aucun resultat d'evaluation disponible pour ce dataset.", status=503)
        return jsonify(result)

    @app.get("/api/fixtures")
    def list_fixtures():
        """Vignettes des images reelles disponibles, groupees par classe."""
        dataset_id = request.args.get("dataset", "mnist")
        try:
            dataset = catalog.dataset(dataset_id)
        except KeyError as exc:
            raise ApiError(str(exc), status=404)

        store = resolve_fixtures(dataset_id)
        grouped = store.indices_by_class()
        payload = {
            str(label): [
                {"index": index, "image": inference.raw_to_png_b64(store.raw(index))}
                for index in indices
            ]
            for label, indices in sorted(grouped.items())
        }
        return jsonify(
            {
                "by_class": payload,
                "count": len(store),
                "class_names": dataset.class_names,
            }
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @app.post("/api/sample")
    def sample():
        """Echantillonne n images depuis la prior N(0, I)."""
        data = _payload()
        entry, adapter = resolve_model(data)
        n = _require_int(data, "n", 1, MAX_BATCH, default=8)
        class_label = resolve_class(data, adapter, required=True)
        seed = _optional_seed(data)

        images = adapter.generate(n, class_label, seed)
        return jsonify(
            {
                "images": inference.batch_to_png_b64(images, adapter),
                "model_id": entry.model_id,
                "class_label": class_label,
                "seed": seed,
            }
        )

    @app.post("/api/decode")
    def decode():
        """Decode un vecteur latent fourni explicitement (curseurs d'exploration)."""
        data = _payload()
        entry, adapter = resolve_model(data)
        class_label = resolve_class(data, adapter, required=True)

        try:
            z = adapter.validate_latent(data.get("z"))
        except ValueError as exc:
            raise ApiError(str(exc))

        images = adapter.decode(z.unsqueeze(0), class_label)
        return jsonify(
            {
                "image": inference.tensor_to_png_b64(images[0], adapter),
                "model_id": entry.model_id,
            }
        )

    @app.post("/api/encode")
    def encode():
        """Encode une image reelle du fixture et renvoie mu (pour initialiser les curseurs)."""
        data = _payload()
        entry, adapter = resolve_model(data)
        store = resolve_fixtures(entry.dataset_id)

        index = _require_int(data, "index", 0, max(len(store) - 1, 0))
        class_label = resolve_class(data, adapter, required=False)
        if adapter.is_conditional and class_label is None:
            class_label = store.label_of(index)

        x = encode_fixture(adapter, store, index)
        mu = adapter.encode(x, class_label)
        return jsonify(
            {
                "z": mu[0].detach().cpu().tolist(),
                "index": index,
                "true_label": store.label_of(index),
                "model_id": entry.model_id,
            }
        )

    @app.post("/api/reconstruct")
    def reconstruct():
        """Encode puis decode une image reelle : montre la perte de reconstruction a l'oeil."""
        data = _payload()
        entry, adapter = resolve_model(data)
        store = resolve_fixtures(entry.dataset_id)

        index = _require_int(data, "index", 0, max(len(store) - 1, 0))
        class_label = resolve_class(data, adapter, required=False)
        if adapter.is_conditional and class_label is None:
            class_label = store.label_of(index)

        x = encode_fixture(adapter, store, index)
        mu = adapter.encode(x, class_label)
        reconstruction = adapter.decode(mu, class_label)

        return jsonify(
            {
                # L'original vient du fixture brut : il ne depend d'aucun modele.
                "original": inference.raw_to_png_b64(store.raw(index)),
                "reconstruction": inference.tensor_to_png_b64(reconstruction[0], adapter),
                "index": index,
                "true_label": store.label_of(index),
                "model_id": entry.model_id,
            }
        )

    @app.post("/api/interpolate")
    def interpolate():
        """Interpole entre deux images reelles dans l'espace latent."""
        data = _payload()
        entry, adapter = resolve_model(data)
        store = resolve_fixtures(entry.dataset_id)

        last = max(len(store) - 1, 0)
        steps = _require_int(data, "steps", 2, MAX_STEPS, default=12)
        source = _require_int(data, "source_index", 0, last)
        target = _require_int(data, "target_index", 0, last)

        class_label = resolve_class(data, adapter, required=False)
        if adapter.is_conditional and class_label is None:
            # Condition figee sur la classe de depart : on observe alors la
            # morphologie encodee dans z, pas le changement de condition.
            class_label = store.label_of(source)

        x_source = encode_fixture(adapter, store, source)
        x_target = encode_fixture(adapter, store, target)

        z_source = adapter.encode(x_source, store.label_of(source) if adapter.is_conditional else None)
        z_target = adapter.encode(x_target, store.label_of(target) if adapter.is_conditional else None)

        images = adapter.interpolate(z_source[0], z_target[0], steps, class_label)

        return jsonify(
            {
                "images": inference.batch_to_png_b64(images, adapter),
                "alphas": inference.alphas_for(steps),
                "source": {
                    "index": source,
                    "label": store.label_of(source),
                    "image": inference.raw_to_png_b64(store.raw(source)),
                },
                "target": {
                    "index": target,
                    "label": store.label_of(target),
                    "image": inference.raw_to_png_b64(store.raw(target)),
                },
                "model_id": entry.model_id,
            }
        )

    @app.post("/api/ablation/compare")
    def ablation_compare():
        """Decode un meme z par tous les modeles d'ablation d'un dataset.

        C'est la vue qui rend l'etude d'ablation tangible : a latent identique,
        on voit directement l'effet de beta sur l'image produite.
        """
        data = _payload()
        dataset_id = data.get("dataset", "mnist")
        try:
            candidates = [m for m in catalog.models(dataset_id) if m.ablation_series]
        except KeyError as exc:
            raise ApiError(str(exc), status=404)

        # Les modeles sont regroupes par serie (vae / cvae) : comparer un VAE
        # et un CVAE a beta different melangerait deux effets distincts.
        series: Dict[str, list] = {}
        for model in candidates:
            series.setdefault(model.ablation_series, []).append(model)

        requested = data.get("series")
        if requested is not None:
            if requested not in series:
                raise ApiError(
                    f"Serie d'ablation inconnue : {requested}. "
                    f"Disponibles : {sorted(series)}.",
                    status=404,
                )
            selected = series[requested]
        else:
            # A defaut, la serie la plus fournie : c'est celle qui illustre le
            # mieux l'effet de beta.
            selected = max(series.values(), key=len, default=[])

        if len(selected) < 2:
            raise ApiError(
                f"Serie d'ablation incomplete pour '{dataset_id}' : "
                f"{len(selected)} modele(s), il en faut au moins 2.",
                status=503,
            )

        selected = sorted(selected, key=lambda m: (m.beta is None, m.beta))
        reference = catalog.adapter(selected[0].model_id)
        seed = _optional_seed(data)

        if data.get("z") is not None:
            try:
                z = reference.validate_latent(data.get("z")).unsqueeze(0)
            except ValueError as exc:
                raise ApiError(str(exc))
        else:
            z = reference.sample_latent(1, seed)

        class_label = None
        if selected[0].conditional:
            class_label = _require_int(data, "class_label", 0, reference.num_conditions - 1, default=0)

        results = []
        for entry in selected:
            adapter = catalog.adapter(entry.model_id)
            if adapter.latent_dim != reference.latent_dim:
                # Comparer des latents de dimensions differentes n'aurait aucun sens.
                continue
            image = adapter.decode(z, class_label if adapter.is_conditional else None)
            results.append(
                {
                    "model_id": entry.model_id,
                    "label": entry.label,
                    "beta": entry.beta,
                    "description": entry.description,
                    "image": inference.tensor_to_png_b64(image[0], adapter),
                }
            )

        return jsonify(
            {
                "results": results,
                "z": z[0].detach().cpu().tolist(),
                "seed": seed,
                "dataset": dataset_id,
                "series": selected[0].ablation_series,
                "available_series": sorted(series),
            }
        )

    # ------------------------------------------------------------------ #
    # Erreurs + service du frontend buildé
    # ------------------------------------------------------------------ #

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        return jsonify({"error": error.message}), error.status

    @app.errorhandler(500)
    def handle_internal_error(error):  # pragma: no cover - filet de securite
        return jsonify({"error": "Erreur interne du serveur."}), 500

    @app.get("/")
    @app.get("/<path:requested_path>")
    def serve_frontend(requested_path: str = ""):
        """Sert le build React en mode demonstration.

        En developpement le build n'existe pas : on renvoie un message explicite
        plutot qu'un 404 muet, pour ne pas laisser croire que l'API est cassee.
        """
        if requested_path.startswith("api/"):
            return jsonify({"error": "Endpoint inconnu."}), 404

        if not FRONTEND_DIST.exists():
            return (
                jsonify(
                    {
                        "message": "API en ligne. Le frontend n'est pas buildé.",
                        "developpement": "Lancer 'npm run dev' dans frontend/ puis ouvrir http://localhost:5173",
                        "demonstration": "Lancer 'make demo' pour builder le frontend et le servir ici.",
                        "api": "/api/health",
                    }
                ),
                200,
            )

        candidate = FRONTEND_DIST / requested_path
        if requested_path and candidate.is_file():
            return send_from_directory(FRONTEND_DIST, requested_path)
        # Fallback SPA : toute autre route est geree cote client.
        return send_from_directory(FRONTEND_DIST, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API de demonstration VAE / CVAE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="cpu, cuda ou auto")
    parser.add_argument("--debug", action="store_true", help="Rechargement a chaud (developpement)")
    args = parser.parse_args()

    application = create_app(device=args.device)
    if args.debug:
        application.run(host=args.host, port=args.port, debug=True)
    else:
        # waitress : serveur WSGI de production, multi-thread, dispo sous Windows.
        from waitress import serve

        print(f"API de demonstration sur http://{args.host}:{args.port}")
        serve(application, host=args.host, port=args.port, threads=4)
