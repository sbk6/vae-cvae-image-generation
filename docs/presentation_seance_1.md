# Présentation de séance — VAE / CVAE sur MNIST

Bonsoir Monsieur,

Nous travaillons sur le sujet **VAE conditionnel pour génération d'images**. Pour cette première étape, nous avons commencé uniquement sur **MNIST** afin de valider toute la chaîne technique avant de passer aux autres jeux de données.

## 1. Ce que nous avons compris du sujet

Le projet demande de construire un modèle qui peut :
- apprendre à reconstruire des images,
- apprendre à générer de nouvelles images,
- contrôler la génération avec une étiquette de classe.

Le modèle principal est le **VAE**. La version contrôlée est le **CVAE**.

## 2. Ce que nous avons fait jusqu'à maintenant

Nous avons réalisé les blocs suivants :
- création de la structure du projet,
- lecture de la configuration depuis des fichiers YAML,
- chargement de MNIST,
- implémentation du VAE,
- implémentation du CVAE,
- calcul de la perte ELBO,
- boucle d'entraînement,
- sauvegarde du meilleur modèle,
- génération d'images de contrôle,
- tests unitaires.

## 2 bis. Où se trouve chaque chose dans le code

Pour pouvoir expliquer le projet simplement en séance, voici la correspondance entre les idées et les fichiers du dépôt.

- `src/training/train.py` : point d'entrée principal pour lancer l'entraînement. Ce fichier lit la configuration YAML, prépare les données, crée le modèle et appelle la boucle d'entraînement.
- `src/training/trainer.py` : contient la fonction `train()` qui pilote l'entraînement, la validation, la sauvegarde du meilleur modèle et l'écriture du fichier `training_log.csv`.
- `src/models/vae.py` : contient le modèle VAE. C'est ici que l'image est encodée, transformée en espace latent, puis reconstruite.
- `src/models/cvae.py` : contient le modèle CVAE. C'est la version conditionnelle, donc le label de classe est ajouté à l'entrée et à la génération.
- `src/losses/elbo.py` : calcule la perte ELBO, c'est-à-dire reconstruction + KL.
- `src/data/datasets.py` : charge MNIST et prépare les `DataLoader` pour l'entraînement, la validation et le test.
- `src/utils/config.py` : lit les fichiers YAML et fusionne les paramètres de configuration avec les arguments de ligne de commande.
- `src/utils/seed.py` : fixe la graine aléatoire pour rendre les essais reproductibles.
- `scripts/generate_cvae_grid.py` : lance un court entraînement de contrôle puis génère une grille d'images conditionnées.
- `scripts/inspect_dataloader.py` : sert à vérifier visuellement que les images du dataset sont bien chargées.
- `reports/figures/` : contient les figures générées.
- `reports/training_log.csv` : contient les pertes mesurées à chaque époque.
- `reports/best_checkpoint.pth` : contient le meilleur modèle sauvegardé.

### Fonctions et rôles importants

- `train()` dans `src/training/trainer.py` : boucle principale d'entraînement.
- `run_epoch()` dans `src/training/trainer.py` : exécute une époque complète, en mode entraînement ou validation.
- `VAE.forward()` dans `src/models/vae.py` : prend une image, l'encode, échantillonne dans l'espace latent, puis reconstruit.
- `CVAE.forward()` dans `src/models/cvae.py` : fait la même chose que le VAE, mais avec la condition de classe.
- `CVAE.sample()` dans `src/models/cvae.py` : génère de nouvelles images à partir d'une classe demandée.

### Ce qu'on peut dire en présentation

On peut expliquer par exemple :

- "Dans `src/models/vae.py`, on a écrit le VAE qui apprend à compresser puis reconstruire l'image."
- "Dans `src/models/cvae.py`, on a ajouté la classe en entrée pour contrôler la génération."
- "Dans `src/training/trainer.py`, on a la boucle qui entraîne le modèle et enregistre les résultats."
- "Dans `scripts/generate_cvae_grid.py`, on génère la planche finale d'images pour voir si le modèle sait produire la bonne classe."

## 2 ter. Pourquoi cette architecture et pourquoi le YAML

### Pourquoi cette architecture du projet ?

Nous avons organisé le projet en plusieurs dossiers parce que cela permet de séparer les responsabilités et de ne pas tout mélanger dans un seul gros fichier.

- `src/data/` gère les données.
- `src/models/` gère les modèles.
- `src/losses/` gère la fonction de perte.
- `src/training/` gère l'entraînement.
- `scripts/` gère les petits outils de contrôle.
- `reports/` garde les résultats.

Cette architecture a été choisie pour trois raisons :
- **comprendre plus facilement** où est chaque partie du projet,
- **modifier plus facilement** une partie sans casser les autres,
- **réutiliser plus facilement** le code sur d'autres datasets, ce qui est important puisque le sujet demande MNIST, Fashion-MNIST et CelebA.

### Pourquoi utiliser des fichiers YAML ?

Le YAML sert à mettre les paramètres du projet dans un fichier séparé du code.

Cela est utile parce que :
- on peut changer le dataset, la seed, le nombre d'epochs ou le `beta` sans modifier le code,
- on garde une trace claire de la configuration utilisée pour chaque expérience,
- on peut comparer plusieurs essais de façon propre,
- le projet reste plus reproductible.

En séance, on peut dire simplement :
"Le YAML nous permet de régler le projet sans toucher au code."

## 3. Répartition du travail dans le groupe

Nous sommes trois, et nous avons choisi une répartition simple, claire et proche de ce que le professeur attend : chaque membre a un rôle principal, mais tout le monde comprend le projet complet et peut challenger le travail des autres.

### Rôle principal de chaque membre

- **BIKOZI Sylvain** — données, socle technique et validation de MNIST : préparation des données, configuration YAML, structure du projet, lancement des entraînements, contrôle des résultats de base.
- **DAVID** — modèles et apprentissage : implémentation du VAE, du CVAE et de la perte ELBO, vérification que le modèle apprend correctement.
- **Monix** — résultats et restitution : génération des figures, interprétation des pertes, rédaction de la documentation, préparation de la présentation orale.

### Répartition pratique par dataset

Comme le sujet nous amène à travailler sur trois jeux de données, nous avons décidé de répartir l'avancement par dataset tout en gardant une logique commune :

- **MNIST** : dataset de départ, déjà utilisé pour valider l'ensemble du pipeline.
- **Fashion-MNIST** : prochain dataset à brancher pour vérifier que le code reste générique.
- **CelebA** : troisième dataset prévu pour tester un cas plus complexe et plus riche en conditions.

### Comment on peut le présenter simplement

On peut dire que chacun a un rôle principal, mais que le travail reste collectif :
- **BIKOZI Sylvain** vérifie que le pipeline de base fonctionne sur MNIST,
- **DAVID** pousse l'implémentation du modèle et de la fonction de perte,
- **Monix** s'occupe de la lecture des résultats et de la présentation finale.

Ensuite, chacun challenge le travail des deux autres pour s'assurer que tout est compris et que le code reste générique.

### Comment on se challenge entre nous

Après chaque partie réalisée, les deux autres membres relisent et challengent le travail :
- on vérifie si le code fonctionne réellement,
- on regarde si les résultats sont compréhensibles,
- on demande si la solution est assez générique pour les autres datasets,
- on corrige ensemble les erreurs ou les points flous avant d'avancer.

Cette manière de travailler permet d'éviter qu'une seule personne fasse tout sans que les autres comprennent.

## 4. Résultats déjà visibles

Les fichiers utiles sont :
- `reports/figures/mnist_real_grid.png` : images réelles de référence,
- `reports/figures/cvae_grid.png` : échantillons générés par le CVAE,
- `reports/figures/cvae_grid_8.png` : version plus large de la grille,
- `reports/training_log.csv` : historique des pertes,
- `reports/best_checkpoint.pth` : modèle sauvegardé.

## 5. Comment interpréter les résultats

- Si les images générées ressemblent à des chiffres, le modèle a appris la distribution du dataset.
- Si le CVAE génère bien la classe demandée, alors la condition fonctionne.
- Si la perte de reconstruction baisse, cela signifie que les images reconstruites deviennent plus proches des vraies images.
- Si la KL divergence reste trop petite, le modèle peut ignorer l'espace latent.

## 6. Difficultés rencontrées

### Difficulté 1 : le projet ne trouvait pas les modules Python
Solution : ajout des fichiers d'initialisation et correction du chemin d'exécution.

### Difficulté 2 : une valeur YAML était mal lue
Solution : normalisation des paramètres numériques dans les fichiers de configuration.

### Difficulté 3 : le test rapide durait trop longtemps
Solution : limitation du nombre d'époques et de batches pour les vérifications de contrôle.

### Difficulté 4 : le CVAE a besoin du label en entrée
Solution : adaptation de la boucle d'entraînement pour transmettre la condition au modèle.

## 7. Ce que nous comptons faire ensuite

- compléter MNIST jusqu'à obtenir une comparaison propre VAE vs CVAE,
- ajouter Fashion-MNIST,
- ajouter CelebA ou un sous-ensemble,
- réaliser l'ablation sur le poids `beta`,
- visualiser l'espace latent,
- faire l'interpolation entre deux images,
- préparer la comparaison qualitative finale.

## 8. Questions possibles du professeur et réponses courtes

### Question : Pourquoi commencer par MNIST ?
Réponse : parce que c'est le jeu de données le plus simple pour vérifier que tout le pipeline fonctionne.

### Question : Quelle est la différence entre VAE et CVAE ?
Réponse : le VAE génère librement, alors que le CVAE génère en tenant compte d'une classe demandée.

### Question : À quoi sert la perte KL ?
Réponse : elle force l'espace latent à rester bien organisé pour pouvoir générer ensuite de nouvelles images.

### Question : Où voir les résultats ?
Réponse : dans `reports/figures/` pour les images et dans `reports/training_log.csv` pour les valeurs numériques.

### Question : Quel est le principal blocage actuel ?
Réponse : la suite du projet dépend maintenant de l'extension aux autres datasets et de l'étude d'ablation.

## 9. Message de synthèse oral possible

"Bonsoir Monsieur, pour notre sujet VAE conditionnel, nous avons d'abord choisi MNIST afin de valider la structure complète du projet. Nous avons mis en place la configuration, le chargement des données, le VAE, le CVAE, la perte ELBO, l'entraînement et les premières visualisations. Nous avons aussi rencontré des difficultés techniques de lecture YAML et adaptation du CVAE à la boucle d'entraînement, mais elles ont été résolues. Les prochaines étapes sont l'extension aux autres datasets, l'ablation sur `beta`, la visualisation de l'espace latent et la comparaison finale entre VAE et CVAE."
