# deployment/

Packaging et déploiement MLflow des modèles VAE et CVAE Fashion-MNIST.

Ce dossier ne réentraîne rien. Il part des checkpoints déjà figés
dans `checkpoints/` et les transforme en endpoints HTTP utilisables
par l'application web (via FastAPI côté collègue).

## Fichiers

- `mlflow_models.py` — wrappers `mlflow.pyfunc.PythonModel` pour le
  VAE et le CVAE. Réutilise `models/vae.py` et `models/cvae.py` sans
  les modifier.
- `register_models.py` — empaquette les checkpoints et les enregistre
  dans le MLflow Model Registry (`FashionMNIST-VAE`, `FashionMNIST-CVAE`).
- `test_serving.py` — teste les endpoints une fois servis (10 classes
  CVAE + déterminisme).

## Étape 1 — installer les dépendances manquantes

```bash
pip install mlflow pillow requests
```

(`torch`, `numpy`, `pandas` sont déjà utilisés dans le projet.)

## Étape 2 — enregistrer les modèles dans le Model Registry

Depuis la racine du projet (`projects/david_fashion_mnist/`) :

```bash
python -m deployment.register_models
```

Cela crée deux entrées dans le Model Registry :

```
FashionMNIST-VAE   version 1
FashionMNIST-CVAE  version 1
```

Vous pouvez les voir avec :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

puis onglet **Models**.

## Étape 3 — servir les modèles

Dans deux terminaux séparés (les deux serveurs doivent tourner en
même temps) :

```bash
mlflow models serve -m "models:/FashionMNIST-VAE/1"  -p 5001 --env-manager local
mlflow models serve -m "models:/FashionMNIST-CVAE/1" -p 5002 --env-manager local
```

`--env-manager local` évite que MLflow recrée un environnement conda
séparé — il réutilise votre environnement Python actuel, où PyTorch
est déjà installé. C'est plus rapide et suffisant pour un projet
étudiant.

## Étape 4 — tester avant de livrer

Dans un troisième terminal, pendant que les deux serveurs tournent :

```bash
python -m deployment.test_serving
```

Si tout est vert, les endpoints sont prêts à être transmis à votre
collègue :

```
VAE  : http://localhost:5001/invocations
CVAE : http://localhost:5002/invocations
```

## Format des requêtes (à donner à votre collègue FastAPI)

**VAE**

```json
POST http://localhost:5001/invocations
{
  "dataframe_split": {
    "columns": ["z"],
    "data": [[null]]
  }
}
```

**CVAE**

```json
POST http://localhost:5002/invocations
{
  "dataframe_split": {
    "columns": ["class", "z"],
    "data": [[7, null]]
  }
}
```

- `"class"` : entier entre 0 et 9 (obligatoire pour le CVAE).
- `"z"` : soit `null` (le serveur tire un vecteur latent aléatoire),
  soit une liste de 16 flottants (pour le slider d'interpolation :
  envoyer deux z et interpoler côté application, ou envoyer un z
  déjà interpolé directement).

**Réponse** (les deux endpoints) :

```json
[
  {
    "image_base64": "iVBORw0KGgoAAAANS...",
    "image_format": "png",
    "width": 28,
    "height": 28,
    "requested_class": 7,
    "requested_class_name": "Sneaker"
  }
]
```

(`requested_class` et `requested_class_name` uniquement pour le CVAE.)

Côté application web, l'image s'affiche directement avec :

```html
<img src="data:image/png;base64,{image_base64}" />
```

## Ce qui reste figé

Comme convenu, ce dossier ne modifie jamais :

```
checkpoints/vae_beta_1_seed42_final.pt
checkpoints/cvae_beta_1_seed42_final.pt
```

Aucun réentraînement, aucun changement de beta ou de seed n'a lieu
ici.