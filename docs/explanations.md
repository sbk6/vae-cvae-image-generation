# Documentation du projet — Explications détaillées (français)

## 1. Objectif du projet
Ce projet implémente des modèles de type Variational Autoencoder (VAE) et Conditional VAE (CVAE) en PyTorch, avec une approche :
- agnostique au jeu de données (les paramètres sont fournis par YAML),
- reproductible (seed propagée partout),
- dépendances minimales.

But concret réalisé : entraîner un VAE/CVAE court sur MNIST en smoke-test et générer des planches d'images (grilles) montrant des échantillons conditionnés.

## 2. Ce qui a été fait (liste "terre à terre")
- Création de l'arborescence du projet (`src/`, `scripts/`, `configs/`, `tests/`, `reports/`).
- Implémentation d'un chargeur de données MNIST générique (`src/data/datasets.py`).
- Implémentation d'un `VAE` (`src/models/vae.py`) et d'un `CVAE` (`src/models/cvae.py`).
- Écriture de la perte ELBO analytique (reconstruction + KL) dans `src/losses/elbo.py`.
- Boucle d'entraînement générique et gestion du mode conditionné (`src/training/trainer.py`).
- CLI d'entraînement (`src/training/train.py`) et script de génération d'une grille de samples conditionnés (`scripts/generate_cvae_grid.py`).
- Tests unitaires basiques pour valider formes et comportements (`tests/`).
- Génération de figures de contrôle dans `reports/figures/` (réelles et samples CVAE).
- Initialisation d'un dépôt Git local et commit des changements sur la branche `feature/sylvain-mnist-core`.

## 3. Arborescence critique (fichiers principaux)
- `configs/` : fichiers YAML décrivant datasets et training (ex. `mnist_vae.yaml`, `mnist_cvae.yaml`).
- `src/data/datasets.py` : construction des dataloaders.
- `src/models/vae.py` : implémentation du VAE.
- `src/models/cvae.py` : implémentation du CVAE (conditionnement générique).
- `src/losses/elbo.py` : fonction `elbo_loss` retournant dictionnaire {loss, reconstruction, kl, beta}.
- `src/training/trainer.py` : boucle d'entraînement, validation, checkpointing.
- `src/training/train.py` : point d'entrée CLI pour l'entraînement.
- `scripts/generate_cvae_grid.py` : entraîne en court (smoke-test) et sauve une grille d'images conditionnées.
- `reports/figures/` : contient `mnist_real_grid.png`, `cvae_grid.png`, `cvae_grid_8.png`.

## 4. Comment exécuter localement (pas-à-pas)
1. Installer les dépendances (virtualenv recommandé) :

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Tests unitaires (rapides) :

```bash
python -m pytest -q
```

3. Lancer un smoke-test d'entraînement VAE :

```bash
python -m src.training.train --config configs/mnist_vae.yaml --smoke-test
```

4. Générer une grille CVAE conditionnée (court entraînement automatique) :

```bash
python scripts/generate_cvae_grid.py --config configs/mnist_cvae.yaml --output reports/figures/cvae_grid.png --samples-per-class 8
```

Les images seront sauvegardées dans `reports/figures/`.

## 5. Explication simple des concepts (pour débutant)
- Autoencodeur (AE) : réseau qui apprend à reproduire son entrée via un goulot d'encodage puis décodage. Utile pour compression et représentation.
- Variational Autoencoder (VAE) : version probabiliste de l'AE. L'encodeur prédit une distribution (moyenne `mu` et variance `sigma^2`), on échantillonne latent `z ~ N(mu, sigma^2)`, puis le décodeur reconstruit. L'entraînement minimise deux termes :
  - la perte de reconstruction (ici MSE entre image et reconstruction),
  - la divergence KL entre la distribution latente et une prior (généralement N(0, I)).
- CVAE : VAE conditionné sur une information (ex. label). On force le modèle à générer des images correspondant à une condition donnée.

## 6. Interprétation des résultats et métriques (très simple)
- ELBO (Evidence Lower Bound) = - (reconstruction_loss + beta * KL)
- Nous rapportons séparément :
  - `reconstruction` : plus petit vaut mieux (meilleure reconstruction),
  - `kl` : donne combien la latente s'écarte de la prior ; si `kl` ~ 0, le modèle n'utilise pas le latent (problème d'effondrement), si trop grand, latente trop informatif et génération peut être mauvaise.
- Exemples d'interprétation :
  - Reconstruction faible + KL modéré → bon équilibre.
  - Reconstruction faible + KL très faible → modèle ignore latente (posterior collapse).
  - Reconstruction élevée (erreur haute) → modèle pas encore entraîné ou architecture insuffisante.

Figures :
- `mnist_real_grid.png` : patch d'images réelles — sert de référence visuelle.
- `cvae_grid.png` / `cvae_grid_8.png` : pour chaque ligne (condition), des échantillons conditionnés. Regardez si la ligne correspondant à la condition "3" ressemble majoritairement à des 3.

## 7. Glossaire (termes traduits)
- Encoder → Encodeur
- Decoder → Décodeur
- Latent space → Espace latent
- Reconstruction loss → Perte de reconstruction
- KL divergence → Divergence de Kullback-Leibler (KL)
- Epoch → Époque (passage complet sur le dataset)
- Batch → Mini-lot
- Seed → Graine aléatoire (pour reproduction)

## 8. Questions / Réponses possibles (FAQ)
Q: Pourquoi le modèle ne génère-t-il pas des chiffres nets ?
R: Smoke-test court : modèle pas entraîné suffisamment. Augmente `epochs` et vérifie `beta` et `latent_dim`.

Q: Comment changer le dataset ?
R: Modifier ou ajouter un fichier YAML dans `configs/` et étendre `src/data/datasets.py` avec une clé `name` correspondante. Le code est conçu pour être agnostique au dataset si le dataset retourne `DatasetInfo`.

Q: Comment savoir si le modèle utilise bien la condition ?
R: Regarde les grilles conditionnées : si les lignes correspondent aux labels, alors oui. On peut aussi mesurer la mutual information empirique entre label et reconstruction (non fourni automatiquement ici).

Q: Comment ré-exécuter un run reproductible ?
R: Assure-toi d'utiliser la même `seed` dans le YAML (`training.seed`) et le même `device` (CPU/GPU). Les résultats seront proches si tout est identique.

## 9. Prochaines étapes recommandées
- Ajouter des scripts d'évaluation qualitatives (interpolation, t-SNE/UMAP des latents).
- Automatiser l'ablation sur `beta` (script prévu `configs/ablation_beta.yaml`).
- Ajouter des notebooks d'exploration pour visualiser latentes et métriques.

---

Pour toute modification, dis-moi si tu veux :
- que je pousse la branche sur un remote (fournis l'URL remote),
- que j'ajoute une section plus pédagogique (exercices pas-à-pas),
- ou que je crée des notebooks explicatifs.
