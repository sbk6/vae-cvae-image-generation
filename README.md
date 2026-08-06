# VAE / CVAE Image Generation — Compte rendu du projet

Bonsoir Monsieur,

Nous travaillons sur le sujet **VAE conditionnel pour génération d'images**. L'idée du projet est simple à résumer : apprendre à un modèle à reconstruire puis à générer des images, d'abord avec un **VAE** classique, puis avec un **CVAE** qui reçoit une étiquette de classe pour contrôler la génération.

À ce stade, nous avons **commencé uniquement sur MNIST**. Les autres jeux de données demandés dans l'énoncé, comme **Fashion-MNIST** et **CelebA**, sont prévus pour la suite.

## Où nous en sommes aujourd'hui

Ce qui est déjà fait :

- mise en place de l'architecture du projet en Python avec PyTorch,
- lecture de la configuration dans des fichiers YAML,
- chargement de MNIST,
- implémentation d'un **VAE**,
- implémentation d'un **CVAE**,
- calcul de la **perte ELBO**,
- boucle d'entraînement avec sauvegarde du meilleur modèle,
- génération d'une grille d'images réelles et d'images produites par le CVAE,
- tests unitaires de base,
- documentation de synthèse en français.

Ce qui reste à faire :

- étendre le code aux autres jeux de données demandés,
- lancer une vraie étude d'ablation sur le poids `beta`,
- visualiser l'espace latent en 2D,
- ajouter l'interpolation entre deux images,
- comparer proprement VAE et CVAE,
- préparer la partie démonstration web si elle est demandée pour la soutenance.

## Répartition des tâches dans le groupe de 3

La répartition suivante est cohérente avec le travail actuel. Si vous avez déjà des noms, vous pouvez remplacer les rôles par les vrais noms.

- **Membre 1** : préparation des données, configuration YAML, scripts d'entraînement.
- **Membre 2** : implémentation des modèles VAE / CVAE et de la perte ELBO.
- **Membre 3** : tests, visualisation, résultats, documentation et présentation.

## Architecture du dépôt

- `configs/` : paramètres du projet en YAML.
- `src/data/` : chargement et préparation des données.
- `src/models/` : modèles `vae.py` et `cvae.py`.
- `src/losses/` : calcul de la perte ELBO.
- `src/training/` : boucle d'entraînement et point d'entrée CLI.
- `src/utils/` : gestion de la configuration et de la graine aléatoire.
- `scripts/` : petits utilitaires de contrôle et de génération de figures.
- `tests/` : tests automatiques.
- `reports/figures/` : images générées.
- `docs/` : explications détaillées.

## Ce que fait le code, en mots simples

### 1. Le VAE

Le VAE prend une image, la compresse dans un **espace latent**, puis essaie de reconstruire l'image d'origine.

Il apprend deux choses :

- à reconstruire correctement l'image,
- à organiser l'espace latent pour qu'on puisse ensuite générer de nouvelles images.

### 2. Le CVAE

Le CVAE fait la même chose, mais avec une **condition**. Ici, la condition est l'étiquette du chiffre MNIST.

Concrètement : si on demande la classe `3`, le modèle doit générer un `3` et pas un autre chiffre.

### 3. La perte ELBO

La perte utilisée est composée de deux parties :

- la **perte de reconstruction** : mesure si l'image reconstruite ressemble à l'image d'entrée,
- la **KL divergence** : mesure si l'espace latent reste bien organisé.

La formule globale est :

ELBO = reconstruction + `beta` × KL

Si `beta` est trop petit, le modèle peut trop se concentrer sur la reconstruction. Si `beta` est trop grand, il peut au contraire trop régulariser et perdre en qualité visuelle.

## Résultats déjà obtenus

Les résultats sont à regarder dans `reports/figures/` et `reports/` :

- `reports/figures/mnist_real_grid.png` : exemples réels de MNIST, pour servir de référence visuelle,
- `reports/figures/cvae_grid.png` : grille générée par le CVAE,
- `reports/figures/cvae_grid_8.png` : version plus large de la grille CVAE,
- `reports/training_log.csv` : historique des pertes par époque,
- `reports/best_checkpoint.pth` : meilleur modèle sauvegardé.

### Comment interpréter ces résultats

- Si la grille des images générées ressemble à des chiffres reconnaissables, cela veut dire que le modèle apprend la distribution des données.
- Si les lignes de la grille correspondent bien aux classes demandées, cela veut dire que le CVAE utilise correctement la condition.
- Si la perte de reconstruction baisse, c'est bon signe.
- Si la KL divergence est proche de zéro tout le temps, cela peut vouloir dire que le modèle n'utilise pas assez l'espace latent.

Important : les chiffres vus pendant le **smoke-test** sont des résultats de contrôle rapide, pas des résultats finaux de soutenance. Ils servent surtout à vérifier que le pipeline fonctionne.

## Difficultés rencontrées et résolution

### Problème 1 : import Python impossible

Au début, les scripts ne trouvaient pas le module `src`.

**Cause** : le projet n'était pas encore vu comme un package Python exécutable directement.

**Solution** : ajout des fichiers `__init__.py` et correction du chemin d'exécution dans les scripts.

### Problème 2 : la valeur `lr` lue comme une chaîne

Le taux d'apprentissage était lu comme du texte au lieu d'un nombre.

**Cause** : la valeur YAML était écrite d'une manière qui pouvait être interprétée comme chaîne.

**Solution** : normalisation de la valeur en `0.001` dans les fichiers YAML.

### Problème 3 : entraînement trop long pour un test rapide

Le smoke-test lançait trop de batches et prenait trop de temps.

**Cause** : le mode de contrôle n'était pas assez court.

**Solution** : limitation du nombre d'époques et du nombre de batches pour les vérifications rapides.

### Problème 4 : intégration du CVAE dans l'entraînement

Le CVAE a besoin du label en entrée, contrairement au VAE simple.

**Cause** : la boucle d'entraînement initiale traitait tous les modèles comme si l'entrée était uniquement l'image.

**Solution** : détection du modèle conditionnel et passage du label au modèle pendant l'entraînement et la validation.

## Traduction des termes techniques

- Encoder : encodeur
- Decoder : décodeur
- Latent space : espace latent
- Reconstruction loss : perte de reconstruction
- KL divergence : divergence de Kullback-Leibler
- Batch : mini-lot
- Epoch : époque, un passage complet sur les données
- Seed : graine aléatoire
- Condition : information imposée au modèle, ici le label de classe

## Questions / réponses possibles pour la séance

### 1. Quel est l'objectif du projet ?
Le but est de générer des images avec un VAE puis un CVAE, pour comparer une génération libre et une génération contrôlée par classe.

### 2. Qu'avez-vous déjà fait concrètement ?
Nous avons mis en place la structure du projet, le chargement MNIST, le VAE, le CVAE, la perte ELBO, l'entraînement, les tests et les premières figures de contrôle.

### 3. Pourquoi commencer par MNIST ?
Parce que MNIST est simple, rapide à entraîner, et permet de valider toute la chaîne technique avant de passer à des images plus difficiles.

### 4. Comment savoir si le CVAE marche ?
On regarde si les images générées correspondent à la classe demandée. Par exemple, une ligne conditionnée sur `7` doit ressembler à des `7`.

### 5. Quelle est la difficulté principale à ce stade ?
La difficulté principale est de bien faire comprendre au modèle la différence entre reconstruire une image et générer selon une classe précise.

### 6. Qu'allez-vous faire ensuite ?
Étendre le projet aux autres datasets, faire l'ablation sur `beta`, visualiser l'espace latent et préparer la comparaison VAE vs CVAE.

## Étapes suivantes prévues

- intégrer Fashion-MNIST,
- préparer CelebA ou un sous-ensemble adapté,
- lancer l'étude d'ablation sur plusieurs valeurs de `beta`,
- produire une visualisation 2D de l'espace latent,
- faire l'interpolation entre deux exemples,
- compléter la comparaison qualitative VAE vs CVAE,
- préparer la démonstration finale.

## Commandes utiles

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

Lancer les tests :

```bash
python -m pytest -q
```

Lancer un contrôle rapide sur MNIST :

```bash
python -m src.training.train --config configs/mnist_vae.yaml --smoke-test
```

Générer la grille CVAE :

```bash
python scripts/generate_cvae_grid.py --config configs/mnist_cvae.yaml --output reports/figures/cvae_grid.png --samples-per-class 8
```

## Documents à consulter

- [Explications détaillées](docs/explanations.md)

