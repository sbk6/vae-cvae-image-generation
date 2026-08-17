"""Tests de l'API de demonstration multi-datasets.

Ces tests utilisent les checkpoints reellement entraines : ils sont ignores
automatiquement si ceux-ci sont absents (depot fraichement clone, ou
checkpoints Fashion-MNIST pas encore deposes).
"""
import base64
import io
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

pytest.importorskip("flask", reason="flask non installe (pip install -r backend/requirements.txt)")

from PIL import Image  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.catalog import Catalog, parse_beta_tag  # noqa: E402

LATENT_DIM = 16


@pytest.fixture(scope="module")
def client():
    app = create_app(device="cpu")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def catalog():
    return Catalog(device="cpu")


def models_for(dataset_id: str):
    """Identifiants disponibles pour un dataset, ou liste vide."""
    try:
        return [model.model_id for model in Catalog(device="cpu").models(dataset_id)]
    except KeyError:
        return []


MNIST_MODELS = models_for("mnist")
FASHION_MODELS = models_for("fashion_mnist")

requires_mnist = pytest.mark.skipif(
    not MNIST_MODELS, reason="Checkpoints MNIST absents : lancer 'make train-vae' / 'make train-cvae'"
)
requires_fashion = pytest.mark.skipif(
    not FASHION_MODELS,
    reason="Checkpoints Fashion-MNIST absents : les deposer dans "
    "projects/david_fashion_mnist/checkpoints/",
)


def first(models, conditional: bool):
    """Premier modele correspondant a la conditionnalite demandee."""
    catalog = Catalog(device="cpu")
    for model_id in models:
        if catalog.entry(model_id).conditional == conditional:
            return model_id
    return None


def decode_png(data_uri: str) -> np.ndarray:
    assert data_uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return np.array(Image.open(io.BytesIO(raw)))


# --------------------------------------------------------------- catalogue


def test_health_repond(client):
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert "torch" in payload


def test_parse_beta_tag_suit_la_convention_de_nommage():
    """Convention de David : le point decimal est supprime dans les noms."""
    assert parse_beta_tag("01") == 0.1
    assert parse_beta_tag("1") == 1.0
    assert parse_beta_tag("4") == 4.0
    assert parse_beta_tag("05") == 0.5
    assert parse_beta_tag("abc") is None


@requires_mnist
def test_les_datasets_sont_annonces(client):
    payload = client.get("/api/datasets").get_json()
    ids = {dataset["id"] for dataset in payload["datasets"]}
    assert "mnist" in ids

    mnist = next(d for d in payload["datasets"] if d["id"] == "mnist")
    assert mnist["class_names"] == [str(index) for index in range(10)]


@requires_fashion
def test_les_classes_fashion_sont_nommees(client):
    payload = client.get("/api/datasets").get_json()
    fashion = next(d for d in payload["datasets"] if d["id"] == "fashion_mnist")
    # Ni des chiffres, ni la liste MNIST : des libelles de vetements.
    assert len(fashion["class_names"]) == 10
    assert fashion["class_names"][0] != "0"


@requires_mnist
def test_les_identifiants_sont_namespaces(client):
    payload = client.get("/api/models?dataset=mnist").get_json()
    assert all(model["id"].startswith("mnist/") for model in payload["models"])


def test_dataset_inconnu_renvoie_404(client):
    assert client.get("/api/models?dataset=nexistepas").status_code == 404


# ---------------------------------------------------- plages de valeurs

@pytest.mark.parametrize(
    "dataset_id, expected_output, expected_input",
    [("mnist", (-1.0, 1.0), (-1.0, 1.0)), ("fashion_mnist", (0.0, 1.0), (0.0, 1.0))],
)
def test_chaque_famille_declare_sa_propre_plage(catalog, dataset_id, expected_output, expected_input):
    """Garde-fou central de l'integration.

    Les modeles de src/ sortent en [-1, 1] (Tanh) et ceux de projects/ en
    [0, 1] (Sigmoid). Confondre les deux produit une image delavee **sans
    lever d'erreur** : c'est le bug silencieux que cette assertion empeche.
    """
    models = catalog.models(dataset_id)
    if not models:
        pytest.skip(f"Aucun checkpoint pour {dataset_id}")

    adapter = catalog.adapter(models[0].model_id)
    assert adapter.output_range == expected_output
    assert adapter.input_range == expected_input


@requires_fashion
def test_les_images_fashion_couvrent_la_dynamique(client):
    """Une sortie Sigmoid mal reinterpretee comme du [-1, 1] serait ecrasee
    dans la moitie haute de l'histogramme. On verifie que le noir reste
    atteignable sur une image reelle du fixture."""
    payload = client.post(
        "/api/reconstruct", json={"model_id": FASHION_MODELS[0], "index": 0}
    ).get_json()
    original = decode_png(payload["original"])
    assert original.min() < 40, "Le fond des images Fashion-MNIST doit rester sombre"
    assert original.max() > 200, "Les images doivent conserver des pixels clairs"


# ---------------------------------------------------------------- generation


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_sample_renvoie_le_bon_nombre_dimages(client, dataset_id):
    models = models_for(dataset_id)
    if not models:
        pytest.skip(f"Aucun checkpoint pour {dataset_id}")

    conditional_id = first(models, conditional=True)
    body = {"model_id": conditional_id or models[0], "n": 5}
    if conditional_id:
        body["class_label"] = 3

    payload = client.post("/api/sample", json=body).get_json()
    assert len(payload["images"]) == 5
    assert decode_png(payload["images"][0]).shape == (28, 28)


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_sample_est_reproductible_avec_une_seed(client, dataset_id):
    models = models_for(dataset_id)
    unconditional = first(models, conditional=False)
    if not unconditional:
        pytest.skip(f"Aucun modele non conditionnel pour {dataset_id}")

    body = {"model_id": unconditional, "n": 4, "seed": 1234}
    first_run = client.post("/api/sample", json=body).get_json()["images"]
    second_run = client.post("/api/sample", json=body).get_json()["images"]
    assert first_run == second_run


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_la_condition_change_l_image(client, dataset_id):
    """Coeur du CVAE : a z identique, changer la classe doit changer la sortie."""
    models = models_for(dataset_id)
    conditional = first(models, conditional=True)
    if not conditional:
        pytest.skip(f"Aucun modele conditionnel pour {dataset_id}")

    z = [0.3] * LATENT_DIM
    a = client.post("/api/decode", json={"model_id": conditional, "z": z, "class_label": 1})
    b = client.post("/api/decode", json={"model_id": conditional, "z": z, "class_label": 8})
    assert a.get_json()["image"] != b.get_json()["image"]


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_decode_est_deterministe(client, dataset_id):
    models = models_for(dataset_id)
    unconditional = first(models, conditional=False)
    if not unconditional:
        pytest.skip(f"Aucun modele non conditionnel pour {dataset_id}")

    body = {"model_id": unconditional, "z": [0.5] * LATENT_DIM}
    a = client.post("/api/decode", json=body).get_json()["image"]
    b = client.post("/api/decode", json=body).get_json()["image"]
    assert a == b


# ------------------------------------------------- reconstruction / latent


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_reconstruction_renvoie_original_et_reconstruit(client, dataset_id):
    models = models_for(dataset_id)
    if not models:
        pytest.skip(f"Aucun checkpoint pour {dataset_id}")

    payload = client.post("/api/reconstruct", json={"model_id": models[0], "index": 0}).get_json()
    assert decode_png(payload["original"]).shape == (28, 28)
    assert decode_png(payload["reconstruction"]).shape == (28, 28)
    assert payload["original"] != payload["reconstruction"]


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_encode_renvoie_un_latent_de_la_bonne_taille(client, dataset_id):
    models = models_for(dataset_id)
    if not models:
        pytest.skip(f"Aucun checkpoint pour {dataset_id}")

    payload = client.post("/api/encode", json={"model_id": models[0], "index": 0}).get_json()
    assert len(payload["z"]) == LATENT_DIM
    assert all(isinstance(value, float) for value in payload["z"])


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_interpolation_borne_sur_les_extremites(client, dataset_id):
    """Les images alpha=0 et alpha=1 doivent egaler les reconstructions des bornes."""
    models = models_for(dataset_id)
    unconditional = first(models, conditional=False)
    if not unconditional:
        pytest.skip(f"Aucun modele non conditionnel pour {dataset_id}")

    source, target = 0, 60
    interpolation = client.post(
        "/api/interpolate",
        json={"model_id": unconditional, "source_index": source, "target_index": target, "steps": 8},
    ).get_json()

    assert len(interpolation["images"]) == 8
    assert interpolation["alphas"][0] == 0.0
    assert interpolation["alphas"][-1] == 1.0

    start = client.post("/api/reconstruct", json={"model_id": unconditional, "index": source}).get_json()
    end = client.post("/api/reconstruct", json={"model_id": unconditional, "index": target}).get_json()
    assert interpolation["images"][0] == start["reconstruction"]
    assert interpolation["images"][-1] == end["reconstruction"]


# ---------------------------------------------------------------- ablation


@pytest.mark.parametrize("dataset_id", ["mnist", "fashion_mnist"])
def test_ablation_couvre_toute_la_serie_de_betas(client, dataset_id):
    """Regression : beta = 1 doit figurer dans la serie.

    Les modeles beta = 1 servent aussi de modeles principaux ; les filtrer sur
    ce critere amputait la comparaison de sa valeur centrale.
    """
    if not models_for(dataset_id):
        pytest.skip(f"Aucun checkpoint pour {dataset_id}")

    response = client.post("/api/ablation/compare", json={"dataset": dataset_id, "seed": 7})
    if response.status_code == 503:
        pytest.skip(f"Pas de serie d'ablation pour {dataset_id}")

    payload = response.get_json()
    betas = [result["beta"] for result in payload["results"]]
    assert len(betas) >= 3
    assert betas == sorted(betas)
    assert 1.0 in betas
    # Un meme z decode par des betas differents doit donner des images differentes.
    assert len({result["image"] for result in payload["results"]}) == len(payload["results"])


@requires_fashion
def test_ablation_permet_de_choisir_la_serie(client):
    payload = client.post(
        "/api/ablation/compare", json={"dataset": "fashion_mnist", "series": "cvae", "class_label": 8}
    ).get_json()
    assert payload["series"] == "cvae"
    assert all(result["model_id"].startswith("fashion/cvae") for result in payload["results"])


@requires_fashion
def test_serie_inconnue_renvoie_404(client):
    response = client.post(
        "/api/ablation/compare", json={"dataset": "fashion_mnist", "series": "inexistante"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------- validation


@requires_mnist
@pytest.mark.parametrize(
    "path, body, expected",
    [
        ("/api/decode", {"model_id": "mnist/vae_main", "z": [0.1, 0.2]}, 400),
        ("/api/decode", {"model_id": "mnist/vae_main", "z": ["a"] * LATENT_DIM}, 400),
        ("/api/decode", {"model_id": "mnist/vae_main", "z": [float("inf")] * LATENT_DIM}, 400),
        ("/api/decode", {"z": [0.0] * LATENT_DIM}, 400),
        ("/api/decode", {"model_id": "inexistant", "z": [0.0] * LATENT_DIM}, 404),
        # Un identifiant non namespace ne doit plus resoudre.
        ("/api/decode", {"model_id": "vae_main", "z": [0.0] * LATENT_DIM}, 404),
        ("/api/sample", {"model_id": "mnist/cvae_main", "n": 2}, 400),
        ("/api/sample", {"model_id": "mnist/cvae_main", "class_label": 99, "n": 2}, 400),
        ("/api/sample", {"model_id": "mnist/vae_main", "n": 10_000}, 400),
        ("/api/sample", {"model_id": "mnist/vae_main", "n": 0}, 400),
        ("/api/encode", {"model_id": "mnist/vae_main", "index": -1}, 400),
        ("/api/interpolate", {"model_id": "mnist/vae_main", "source_index": 0, "target_index": 1, "steps": 500}, 400),
    ],
)
def test_les_payloads_invalides_renvoient_une_erreur_lisible(client, path, body, expected):
    response = client.post(path, json=body)
    assert response.status_code == expected
    assert "error" in response.get_json()


@requires_mnist
def test_corps_non_json_rejete(client):
    response = client.post("/api/decode", data="pas du json", content_type="text/plain")
    assert response.status_code == 400


def test_route_api_inconnue_renvoie_du_json(client):
    response = client.get("/api/nexiste_pas")
    assert response.status_code == 404
    assert "error" in response.get_json()
