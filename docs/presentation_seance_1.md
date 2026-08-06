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

## 3. Répartition du travail dans le groupe

Nous sommes trois, et la répartition peut être présentée ainsi :
- personne 1 : données et configuration,
- personne 2 : modèles et fonction de perte,
- personne 3 : tests, résultats, documentation et préparation de la présentation.

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

"Bonsoir Monsieur, pour notre sujet VAE conditionnel, nous avons d'abord choisi MNIST afin de valider la structure complète du projet. Nous avons mis en place la configuration, le chargement des données, le VAE, le CVAE, la perte ELBO, l'entraînement et les premières visualisations. Nous avons aussi rencontré des difficultés techniques de type import Python, lecture YAML et adaptation du CVAE à la boucle d'entraînement, mais elles ont été résolues. Les prochaines étapes sont l'extension aux autres datasets, l'ablation sur `beta`, la visualisation de l'espace latent et la comparaison finale entre VAE et CVAE."
