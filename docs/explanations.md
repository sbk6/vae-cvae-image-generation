# Documentation du projet — Explications détaillées (français)

> Document technique complémentaire au [README.md](../README.md). Le README est le document de présentation (résultats, interprétation, difficultés) ; ce document explique le fonctionnement du code plus en détail et sert de référence pour l'exécuter soi-même.

## 1. Objectif du projet
Ce projet implémente des modèles de type Variational Autoencoder (VAE) et Conditional VAE (CVAE) en PyTorch, avec une approche :
- agnostique au jeu de données (les paramètres sont fournis par YAML),
- reproductible (seed propagée partout),
- dépendances minimales.

But concret réalisé à ce stade : entraîner un VAE et un CVAE **complets** (10 epochs, données complètes, pas un simple smoke-test) sur MNIST, mener une étude d'ablation sur le poids `beta`, visualiser l'espace latent, produire une interpolation, et comparer quantitativement les deux modèles.

## 2. Ce qui a été fait (liste "terre à terre")
- Arborescence du projet (`src/`, `scripts/`, `configs/`, `tests/`, `reports/`).
- Chargeur de données MNIST générique (`src/data/datasets.py`), avec support d'un sous-échantillonnage optionnel du train set (`dataset.train_subset`) utilisé pour accélérer l'étude d'ablation.
- Implémentation d'un `VAE` (`src/models/vae.py`) et d'un `CVAE` (`src/models/cvae.py`).
- Perte ELBO analytique (reconstruction + KL) dans `src/losses/elbo.py`.
- Boucle d'entraînement générique et gestion du mode conditionné (`src/training/trainer.py`), avec un dossier de sortie dédié par expérience (`training.output_dir`) pour éviter qu'un entraînement en écrase un autre.
- CLI d'entraînement unique (`src/training/train.py`) qui instancie un VAE ou un CVAE selon `model.type` dans le YAML.
- Script d'étude d'ablation (`scripts/run_ablation.py`) : entraîne plusieurs VAE avec des valeurs de `beta` différentes, produit un tableau Markdown (`docs/RESULTATS.md`) et une courbe.
- Visualisation de l'espace latent en 2D par t-SNE (`src/visualization/latent.py`).
- Interpolation entre deux exemples dans l'espace latent (`src/visualization/interpolation.py`).
- Scripts de génération de grilles à partir d'un modèle **déjà entraîné** (`scripts/generate_cvae_grid.py`, `scripts/generate_vae_recon_grid.py`) — ils ne réentraînent plus le modèle à chaque appel.
- Script de comparaison quantitative VAE vs CVAE (`scripts/evaluate.py`), avec une mesure de contrôlabilité pour le CVAE.
- Suivi des expériences avec **MLflow** (`training.mlflow` dans le YAML) : paramètres, métriques par epoch et artefacts (checkpoint, log CSV) enregistrés automatiquement pendant l'entraînement. Les runs déjà réalisés avant l'ajout de MLflow ont été réimportés avec `scripts/backfill_mlflow.py`. Détails et commandes dans le README, section 9.
- Tests unitaires (8 tests, tous verts) pour valider les formes et comportements de base.

## 3. Arborescence critique (fichiers principaux)
- `configs/mnist_vae.yaml`, `configs/mnist_cvae.yaml` : configuration des modèles principaux (10 epochs, données complètes, `beta=1.0`).
- `configs/ablation_beta.yaml` : configuration de l'étude d'ablation (3 valeurs de `beta`, 6 epochs, sous-ensemble de 12 000 images).
- `src/data/datasets.py` : construction des dataloaders. Contient les points d'extension pour Fashion-MNIST et CelebA (non implémentés à ce stade — voir README section "Limites actuelles").
- `src/models/vae.py` : implémentation du VAE.
- `src/models/cvae.py` : implémentation du CVAE (conditionnement générique `one_hot` / `multi_label`).
- `src/losses/elbo.py` : fonction `elbo_loss` retournant `{loss, reconstruction, kl, beta}`.
- `src/training/trainer.py` : boucle d'entraînement, validation, checkpointing, logs CSV.
- `src/training/train.py` : point d'entrée CLI pour l'entraînement (VAE ou CVAE selon la config).
- `src/visualization/common.py` : fonctions partagées pour reconstruire un modèle depuis sa config et charger un checkpoint.
- `src/visualization/latent.py`, `src/visualization/interpolation.py` : visualisations de l'espace latent.
- `scripts/run_ablation.py` : étude d'ablation sur `beta`.
- `scripts/generate_cvae_grid.py`, `scripts/generate_vae_recon_grid.py` : génération de figures à partir d'un checkpoint existant.
- `scripts/evaluate.py` : comparaison quantitative VAE vs CVAE.
- `scripts/backfill_mlflow.py` : réimporte dans MLflow les entraînements déjà réalisés avant que le tracking ne soit branché dans `trainer.py`.
- `reports/experiments/<nom>/` : un dossier par expérience (`vae_main`, `cvae_main`, `ablation/beta_0.1`, etc.), contenant `training_log.csv` et `best_checkpoint.pth`.
- `reports/figures/` : toutes les figures produites — voir le tableau détaillé dans le README, section 6.
- `mlflow.db` : base SQLite locale contenant l'historique des runs MLflow (paramètres, métriques, liens vers les artefacts).

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

3. Entraîner le VAE principal (≈30-45 min sur CPU, pas de smoke-test) :

```bash
python -m src.training.train --config configs/mnist_vae.yaml
```

4. Entraîner le CVAE principal :

```bash
python -m src.training.train --config configs/mnist_cvae.yaml
```

Un test rapide (quelques batches, pour vérifier que le code tourne sans attendre un entraînement complet) reste possible avec `--smoke-test` :

```bash
python -m src.training.train --config configs/mnist_vae.yaml --smoke-test
```

5. Lancer l'étude d'ablation sur `beta` (≈15-25 min sur CPU) :

```bash
python scripts/run_ablation.py --config configs/ablation_beta.yaml
```

6. Générer les grilles d'images à partir d'un modèle déjà entraîné (rapide, pas de réentraînement) :

```bash
python scripts/generate_vae_recon_grid.py --checkpoint reports/experiments/vae_main/best_checkpoint.pth
python scripts/generate_cvae_grid.py --checkpoint reports/experiments/cvae_main/best_checkpoint.pth --samples-per-class 8
```

7. Visualiser l'espace latent et une interpolation :

```bash
python -m src.visualization.latent --config configs/mnist_vae.yaml --checkpoint reports/experiments/vae_main/best_checkpoint.pth --output reports/figures/latent_tsne_vae.png
python -m src.visualization.interpolation --config configs/mnist_vae.yaml --checkpoint reports/experiments/vae_main/best_checkpoint.pth --output reports/figures/interpolation_vae_3_to_8.png --class-a 3 --class-b 8
```

8. Comparaison quantitative VAE vs CVAE :

```bash
python scripts/evaluate.py
```

9. Consulter le suivi des expériences dans MLflow (params, métriques, artefacts de chaque run) :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
puis ouvrir `http://127.0.0.1:5000`. MLflow est activé par défaut dans nos configs (`training.mlflow.enabled: true`) : tout nouvel entraînement lancé avec les commandes ci-dessus y apparaît automatiquement, sans commande supplémentaire.

Toutes les images sont sauvegardées dans `reports/figures/`.

## 5. Explication simple des concepts (pour débutant)
- **Autoencodeur (AE)** : réseau qui apprend à reproduire son entrée via un goulot d'encodage puis décodage. Utile pour compression et représentation.
- **Variational Autoencoder (VAE)** : version probabiliste de l'AE. L'encodeur prédit une distribution (moyenne `mu` et variance `sigma^2`), on échantillonne un latent `z ~ N(mu, sigma^2)` (via la "reparamétrisation" : `z = mu + eps * sigma`, avec `eps` tiré au hasard, ce qui permet de rétropropager le gradient malgré le tirage aléatoire), puis le décodeur reconstruit. L'entraînement minimise deux termes :
  - la perte de reconstruction (MSE entre image et reconstruction),
  - la divergence KL entre la distribution latente apprise et une loi normale standard `N(0, I)` (le "prior").
- **CVAE** : VAE conditionné sur une information (ici, le label de classe). Le label est concaténé à l'image en entrée de l'encodeur, et au vecteur latent en entrée du décodeur. Cela force le modèle à générer des images correspondant à la condition donnée, sans que l'espace latent ait besoin d'encoder lui-même l'identité de la classe.
- **β (beta)** : poids appliqué au terme KL. Un `beta` élevé force un espace latent très proche de `N(0, I)`, au prix d'une reconstruction moins fidèle (et, en excès, d'un effondrement de l'espace latent, voir "posterior collapse" ci-dessous). Un `beta` faible privilégie la reconstruction, au prix d'un espace latent moins régulier.
- **Posterior collapse** : quand `beta` est trop élevé, le modèle "abandonne" et arrête d'utiliser l'espace latent (KL proche de 0) : le décodeur produit presque toujours la même image, quel que soit `z`. Nous l'avons observé concrètement avec `beta=5.0` dans l'étude d'ablation (voir README, section 7).

## 6. Interprétation des résultats et métriques
- ELBO (Evidence Lower Bound), minimisée pendant l'entraînement : `loss = reconstruction + beta * KL`.
- `reconstruction` : plus petit vaut mieux (meilleure reconstruction).
- `kl` : mesure l'écart entre la distribution latente apprise et le prior. Si `kl` ~ 0, le modèle n'utilise pas le latent (posterior collapse) ; si trop grand, le latent est peu régularisé et la génération à partir d'un `z` aléatoire risque d'être mauvaise.
- Grille d'interprétation :
  - Reconstruction faible + KL modéré → bon équilibre (c'est notre cas avec `beta=1.0`, KL≈15-17).
  - Reconstruction faible + KL très faible → modèle ignore le latent (observé avec `beta=5.0`, KL≈0.56).
  - Reconstruction élevée (erreur haute) et KL très élevé → modèle pas assez entraîné, ou `beta` trop faible (observé avec `beta=0.1`, KL≈39.5 : le modèle privilégie fortement la reconstruction au détriment de la régularité du latent).

Les chiffres réels obtenus cette séance (entraînements complets, pas des smoke-tests) sont détaillés dans le [README.md](../README.md), sections 6 et 7. Ce document-ci reste volontairement générique pour rester valable après de futurs runs.

## 7. Glossaire (termes traduits)
| Terme anglais | Traduction / explication |
|---|---|
| Encoder | Encodeur |
| Decoder | Décodeur |
| Latent space | Espace latent |
| Reconstruction loss | Perte de reconstruction |
| KL divergence | Divergence de Kullback-Leibler (KL) |
| Epoch | Époque (un passage complet sur le dataset) |
| Batch | Mini-lot |
| Seed | Graine aléatoire (pour reproduction) |
| Beta (β-VAE) | Poids du terme KL dans la loss |
| Posterior collapse | Effondrement de l'espace latent (le modèle cesse de l'utiliser) |
| Checkpoint | Sauvegarde des poids du modèle |

## 8. Questions / Réponses possibles (FAQ)

**Q : Pourquoi les images générées sont-elles floues ?**
R : C'est une propriété connue des VAE entraînés avec une perte MSE (par opposition, par exemple, à un GAN) : le modèle a tendance à "moyenner" les incertitudes plutôt que de trancher nettement. Ce n'est pas un bug.

**Q : Comment changer le dataset ?**
R : Ajouter/modifier un fichier YAML dans `configs/` et étendre `src/data/datasets.py` avec la clé `name` correspondante. Le code est conçu pour être agnostique au dataset tant que le loader retourne un `DatasetInfo`. Attention : à ce stade, `fashion_mnist` est déclaré dans le code mais **charge encore MNIST** en interne — ce n'est pas encore un vrai second dataset (voir README, section "Limites actuelles").

**Q : Comment savoir si le CVAE utilise bien la condition ?**
R : Regarder `reports/figures/cvae_grid.png` — chaque ligne doit correspondre à la classe demandée. Nous avons aussi essayé une mesure automatique (plus proche centroïde), dont les limites sont expliquées en détail dans le README, section 6.4 : à utiliser avec prudence, l'inspection visuelle reste plus fiable à ce stade.

**Q : Comment ré-exécuter un run reproductible ?**
R : Utiliser la même `seed` dans le YAML (`training.seed`) et le même `device`. Les résultats seront identiques (ou très proches) si tout le reste ne change pas.

**Q : Comment MLflow s'articule avec `training_log.csv` ? Il faut choisir entre les deux ?**
R : Non, les deux coexistent : `training_log.csv` reste écrit à chaque entraînement (simple, lisible sans dépendance), et MLflow enregistre exactement les mêmes métriques en plus, mais dans une interface consultable et comparable entre runs. Si MLflow est désactivé dans le YAML (`training.mlflow.enabled: false`), seul le CSV est produit — le code fonctionne à l'identique.

**Q : Pourquoi limiter l'ablation à un sous-ensemble de données ?**
R : Contrainte de temps de calcul (CPU uniquement, un epoch complet prend ~2min30 à 3min sur les 54 000 images). Le sous-échantillonnage ne s'applique qu'au train set : la validation reste complète, donc la comparaison entre les valeurs de `beta` reste fiable.

## 9. Prochaines étapes recommandées
Voir le README, section 11, pour la liste complète et priorisée (Fashion-MNIST, CelebA, classifieur pour la contrôlabilité, démo web, FID).
