"""
Teste automatiquement les endpoints MLflow avant de les transmettre
à l'équipe FastAPI / application web.

Prérequis
---------
Les deux serveurs MLflow doivent déjà tourner, par exemple :

    mlflow models serve -m "models:/FashionMNIST-VAE/1"  -p 5001 --env-manager local
    mlflow models serve -m "models:/FashionMNIST-CVAE/1" -p 5002 --env-manager local

Utilisation
-----------
    python -m deployment.test_serving

Ce script vérifie :

1. VAE :
   - le endpoint répond (HTTP 200) ;
   - l'image renvoyée est un PNG base64 valide de 28x28.

2. CVAE :
   - pour chacune des 10 classes Fashion-MNIST :
       - le endpoint répond ;
       - l'image renvoyée est valide ;
       - "requested_class" dans la réponse correspond à la classe
         demandée.

3. CVAE avec un z fixé (utile pour le slider d'interpolation de
   l'application) :
   - deux requêtes avec le même z et la même classe renvoient
     exactement la même image (déterminisme).

Le script échoue bruyamment (exceptions explicites) si un test rate,
plutôt que d'afficher un résumé silencieux -- l'objectif est de
détecter un problème AVANT de livrer les endpoints.
"""

from __future__ import annotations

import base64
import io

import requests
from PIL import Image

VAE_ENDPOINT = "http://localhost:5001/invocations"
CVAE_ENDPOINT = "http://localhost:5002/invocations"

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

REQUEST_TIMEOUT_SECONDS = 30


def _post_dataframe_split(url: str, columns: list[str], data: list[list]) -> list[dict]:
    """
    Envoie une requête au format MLflow "dataframe_split" et renvoie
    la liste de résultats.
    """

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": data,
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Le endpoint {url} a répondu avec le statut "
            f"{response.status_code} : {response.text}"
        )

    body = response.json()

    # Selon la version de MLflow, la réponse est soit la liste
    # directement, soit encapsulée sous la clé "predictions".
    if isinstance(body, dict) and "predictions" in body:
        return body["predictions"]

    return body


def _validate_generated_image(result: dict) -> Image.Image:
    """
    Vérifie qu'un résultat contient bien un PNG base64 valide de
    28x28 pixels, et renvoie l'image décodée.
    """

    if "image_base64" not in result:
        raise AssertionError(
            f"Champ 'image_base64' manquant dans la réponse : {result}"
        )

    raw_bytes = base64.b64decode(result["image_base64"])
    image = Image.open(io.BytesIO(raw_bytes))

    if image.size != (28, 28):
        raise AssertionError(
            f"Dimensions inattendues : {image.size}, attendu (28, 28)."
        )

    return image


def test_vae_endpoint() -> None:
    """
    Vérifie que le endpoint VAE répond et génère une image valide.
    """

    print("Test du endpoint VAE ...")

    results = _post_dataframe_split(
        url=VAE_ENDPOINT,
        columns=["z"],
        data=[[None]],
    )

    assert len(results) == 1, "Une seule image attendue."

    _validate_generated_image(results[0])

    print("  -> OK : le VAE génère une image 28x28 valide.")


def test_cvae_endpoint_all_classes() -> None:
    """
    Vérifie que le CVAE génère correctement une image pour chacune
    des 10 classes Fashion-MNIST.
    """

    print("Test du endpoint CVAE sur les 10 classes ...")

    for class_id, class_name in enumerate(FASHION_MNIST_CLASSES):

        results = _post_dataframe_split(
            url=CVAE_ENDPOINT,
            columns=["class", "z"],
            data=[[class_id, None]],
        )

        assert len(results) == 1, "Une seule image attendue."

        result = results[0]

        _validate_generated_image(result)

        if result.get("requested_class") != class_id:
            raise AssertionError(
                f"Classe demandée {class_id}, mais la réponse indique "
                f"{result.get('requested_class')}."
            )

        print(f"  -> OK : classe {class_id} ({class_name})")


def test_cvae_deterministic_with_fixed_z() -> None:
    """
    Vérifie que fournir le même z et la même classe produit deux
    fois exactement la même image.

    C'est important pour le slider d'interpolation de l'application :
    l'application doit pouvoir régénérer une image identique en
    renvoyant le même z.
    """

    print("Test du déterminisme du CVAE avec un z fixé ...")

    fixed_z = [0.1] * 16

    results_first = _post_dataframe_split(
        url=CVAE_ENDPOINT,
        columns=["class", "z"],
        data=[[3, fixed_z]],
    )

    results_second = _post_dataframe_split(
        url=CVAE_ENDPOINT,
        columns=["class", "z"],
        data=[[3, fixed_z]],
    )

    if results_first[0]["image_base64"] != results_second[0]["image_base64"]:
        raise AssertionError(
            "Le CVAE devrait produire exactement la même image "
            "lorsque 'class' et 'z' sont identiques."
        )

    print("  -> OK : même z + même classe => image identique.")


def main() -> None:

    print("=" * 74)
    print("TEST DES ENDPOINTS MLFLOW AVANT LIVRAISON")
    print("=" * 74)

    test_vae_endpoint()
    test_cvae_endpoint_all_classes()
    test_cvae_deterministic_with_fixed_z()

    print("=" * 74)
    print("TOUS LES TESTS SONT PASSÉS.")
    print("Les endpoints peuvent être transmis à l'application web :")
    print(f"  VAE  : {VAE_ENDPOINT}")
    print(f"  CVAE : {CVAE_ENDPOINT}")
    print("=" * 74)


if __name__ == "__main__":
    main()