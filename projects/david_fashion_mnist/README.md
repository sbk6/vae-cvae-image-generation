# VAE et CVAE sur Fashion-MNIST

## Contribution Fashion-MNIST — David

Ce sous-projet étudie l'utilisation des **Variational Autoencoders (VAE)** et des **Conditional Variational Autoencoders (CVAE)** pour la reconstruction et la génération d'images du jeu de données **Fashion-MNIST**.

Le travail réalisé comprend :

- l'implémentation d'un VAE convolutionnel ;
- l'implémentation d'un CVAE convolutionnel conditionné par les classes ;
- l'entraînement des modèles sur Fashion-MNIST ;
- une étude d'ablation du coefficient `beta` ;
- l'évaluation quantitative sur les 10 000 images du jeu officiel de test ;
- l'analyse qualitative des reconstructions et des générations ;
- la visualisation de l'espace latent avec PCA et UMAP ;
- l'étude d'interpolations dans l'espace latent ;
- la sauvegarde des résultats expérimentaux pour assurer la reproductibilité.

L'intégration de **MLflow** et d'un mécanisme d'**early stopping** constitue la prochaine étape du pipeline expérimental.

---

## 1. Jeu de données

Le projet utilise **Fashion-MNIST**, composé d'images en niveaux de gris de taille `28 × 28`.

Le jeu contient :

- 60 000 images officielles d'entraînement ;
- 10 000 images officielles de test ;
- 10 classes de vêtements et accessoires.

Les 60 000 images d'entraînement sont séparées de manière reproductible en :

- 54 000 images pour l'entraînement ;
- 6 000 images pour la validation.

Les 10 000 images officielles de test sont conservées séparément et ne sont utilisées qu'après l'entraînement pour l'évaluation finale.

La séparation train/validation utilise la graine aléatoire :

```text
seed = 42
```

### Classes Fashion-MNIST

| Classe | Nom |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Les images sont converties en tenseurs et les valeurs des pixels sont placées dans l'intervalle `[0, 1]`.

---

## 2. Modèles étudiés

Deux modèles ont été implémentés.

### 2.1 VAE

Le **Variational Autoencoder** apprend une représentation latente probabiliste des images.

L'encodeur convolutionnel transforme une image Fashion-MNIST en deux vecteurs :

- `mu` : moyenne de la distribution latente ;
- `logvar` : logarithme de la variance.

Un vecteur latent `z` est ensuite obtenu avec l'astuce de reparamétrisation :

```text
z = mu + epsilon * sigma
```

avec :

```text
epsilon ~ N(0, I)
```

Le décodeur reconstruit ensuite l'image à partir du vecteur latent.

Architecture principale :

```text
Image 1 × 28 × 28
        |
        v
Conv2d 1 -> 32
        |
        v
Conv2d 32 -> 64
        |
        v
Flatten
        |
        v
Couche cachée : 256
        |
        +------> mu
        |
        +------> logvar
                  |
                  v
           espace latent
             dimension 16
                  |
                  v
              décodeur
                  |
                  v
         image reconstruite
```

La sortie du décodeur utilise une fonction `Sigmoid`, adaptée aux pixels dans `[0, 1]`.

---

### 2.2 CVAE

Le **Conditional Variational Autoencoder** reprend le principe du VAE mais ajoute l'information de classe.

Pour Fashion-MNIST, les labels sont représentés sous forme one-hot sur 10 classes.

La condition est utilisée :

- pendant l'encodage ;
- pendant le décodage.

Le CVAE permet donc de demander explicitement la génération d'une classe particulière.

Par exemple :

```text
classe 5 -> Sandal
classe 8 -> Bag
classe 9 -> Ankle boot
```

Cette propriété constitue l'un des principaux intérêts du CVAE par rapport au VAE classique.

---

## 3. Fonction de perte

Les modèles utilisent une fonction objectif de type beta-VAE :

```text
Loss = Reconstruction + beta × KL
```

Le terme de reconstruction mesure la différence entre l'image originale et l'image reconstruite.

Dans cette implémentation, la reconstruction utilise la **Binary Cross Entropy (BCE)**.

Le terme KL correspond à la divergence de Kullback-Leibler entre la distribution latente apprise et une distribution normale standard :

```text
N(0, I)
```

Le coefficient `beta` contrôle le compromis entre :

- fidélité de reconstruction ;
- régularisation de l'espace latent.

Trois valeurs ont été étudiées :

```text
beta = 0.1
beta = 1
beta = 4
```

---

## 4. Configuration expérimentale

Les six expériences principales utilisent les mêmes paramètres afin de permettre une comparaison cohérente.

| Paramètre | Valeur |
|---|---:|
| Dataset | Fashion-MNIST |
| Train | 54 000 images |
| Validation | 6 000 images |
| Test | 10 000 images |
| Taille des images | 28 × 28 |
| Batch size | 128 |
| Dimension latente | 16 |
| Dimension cachée | 256 |
| Learning rate | 0.001 |
| Optimiseur | Adam |
| Seed | 42 |
| Époques | 20 |

Les modèles étudiés sont :

```text
VAE  beta = 0.1
VAE  beta = 1
VAE  beta = 4

CVAE beta = 0.1
CVAE beta = 1
CVAE beta = 4
```

À chaque époque, les performances sont mesurées sur le jeu de validation.

Le meilleur checkpoint est déterminé à partir de la plus faible loss totale de validation.

---

## 5. Résultats quantitatifs

L'évaluation finale est effectuée sur les **10 000 images du jeu officiel de test Fashion-MNIST**.

Ces images n'ont pas été utilisées pour modifier les poids des modèles.

### Résultats sur le jeu de test

| Modèle | beta | Meilleure époque | Test total | Reconstruction | KL |
|---|---:|---:|---:|---:|---:|
| VAE | 0.1 | 20 | 216.7850 | 212.6656 | 41.1930 |
| VAE | 1 | 20 | 238.3688 | 224.0078 | 14.3610 |
| VAE | 4 | 20 | 267.2967 | 240.0214 | 6.8188 |
| CVAE | 0.1 | 20 | 216.7171 | 212.7228 | 39.9435 |
| CVAE | 1 | 20 | 235.6971 | 224.3014 | 11.3957 |
| CVAE | 4 | 20 | 258.3501 | 241.1858 | 4.2911 |

Les données complètes sont disponibles dans :

```text
results/evaluation_metrics.csv
```

### Remarque importante sur la loss totale

La loss totale dépend directement de la valeur de `beta` :

```text
Loss = Reconstruction + beta × KL
```

Il n'est donc pas méthodologiquement correct de comparer directement les valeurs de loss totale obtenues avec des valeurs de `beta` différentes comme s'il s'agissait exactement du même objectif.

L'analyse entre différentes valeurs de `beta` repose principalement sur :

- la reconstruction ;
- le terme KL ;
- la qualité visuelle des images ;
- l'organisation de l'espace latent.

---

## 6. Étude d'ablation de beta

L'étude d'ablation met clairement en évidence le compromis classique du beta-VAE.

Lorsque `beta` augmente :

```text
beta augmente
      |
      +--> KL diminue
      |
      +--> contrainte sur l'espace latent augmente
      |
      +--> reconstruction devient moins précise
```

Les résultats obtenus montrent cette tendance pour le VAE comme pour le CVAE.

### beta = 0.1

La reconstruction est la meilleure parmi les trois valeurs testées.

En revanche, le terme KL est élevé, ce qui indique que la distribution latente reste relativement éloignée du prior `N(0, I)`.

### beta = 1

Cette valeur fournit le compromis qualitatif le plus équilibré entre :

- reconstruction ;
- régularisation ;
- génération ;
- diversité des images.

### beta = 4

La régularisation est beaucoup plus forte.

Le terme KL diminue fortement, mais les reconstructions deviennent plus lissées et les générations tendent davantage vers des prototypes de classes.

Les graphiques de l'étude sont disponibles dans :

```text
results/ablation/
```

---

## 7. Analyse des reconstructions

Les reconstructions permettent d'observer la quantité d'information conservée après le passage dans l'espace latent.

Les résultats montrent globalement :

### beta = 0.1

Les détails visuels des images originales sont mieux conservés.

### beta = 1

Les reconstructions restent reconnaissables tout en utilisant un espace latent davantage régularisé.

### beta = 4

Les images deviennent plus lissées et certaines caractéristiques fines disparaissent.

Les différences entre VAE et CVAE sont relativement modestes concernant uniquement la reconstruction.

L'intérêt principal du CVAE apparaît davantage lors de la génération conditionnelle.

Les figures sont disponibles dans :

```text
results/reconstructions/
```

---

## 8. Analyse des générations

### VAE

Le VAE génère des images sans imposer de classe particulière.

Avec `beta = 0.1`, les images présentent une bonne diversité mais certaines générations sont moins stables.

Avec `beta = 1`, les générations présentent un meilleur équilibre entre structure et diversité.

Avec `beta = 4`, les générations sont souvent plus simples et moins diversifiées.

### CVAE

Le CVAE permet de générer des images en imposant une classe Fashion-MNIST.

#### CVAE beta = 0.1

Malgré de bonnes performances de reconstruction, les générations conditionnelles sont relativement instables.

Certaines images présentent des confusions entre classes.

Une interprétation possible est que la faible régularisation permet à `z` de conserver beaucoup d'information, y compris de l'information liée à la classe. Le modèle peut alors dépendre davantage du vecteur latent et moins fortement du label fourni au décodeur.

#### CVAE beta = 1

Ce modèle fournit le meilleur compromis qualitatif parmi les trois CVAE étudiés :

- classes généralement reconnaissables ;
- cohérence des générations ;
- diversité satisfaisante.

#### CVAE beta = 4

Les classes générées sont généralement très cohérentes, mais les images deviennent davantage répétitives et proches de prototypes.

Les figures sont disponibles dans :

```text
results/generations/
```

---

## 9. Visualisation de l'espace latent

L'organisation de l'espace latent a été étudiée avec deux méthodes de réduction de dimension :

- PCA ;
- UMAP.

Les visualisations utilisent les vecteurs `mu` produits par l'encodeur plutôt que des échantillons stochastiques `z`.

Cela permet d'obtenir une représentation déterministe de chaque image.

Un sous-ensemble équilibré de **3 000 images**, soit environ 300 images par classe, est utilisé pour les visualisations.

Le même échantillon et la même graine sont utilisés pour tous les modèles afin de permettre une comparaison cohérente.

### PCA

PCA fournit une projection linéaire de l'espace latent de dimension 16 vers deux dimensions.

### UMAP

UMAP permet d'observer davantage la structure locale de l'espace latent.

Les graphiques montrent notamment que :

- avec une faible valeur de `beta`, le VAE conserve davantage de structure associée aux classes ;
- lorsque `beta` augmente, les distributions latentes deviennent davantage régularisées ;
- dans le CVAE, un mélange plus important des classes dans `z` n'est pas nécessairement un échec, puisque le label de classe est fourni séparément au modèle.

Les projections PCA et UMAP restent cependant des représentations en deux dimensions d'un espace latent de dimension 16 et ne doivent pas être surinterprétées.

Les résultats sont disponibles dans :

```text
results/latent_spaces/
```

---

## 10. Interpolations latentes

Des interpolations ont été réalisées entre deux images appartenant à la même classe.

Deux images réelles sont encodées pour obtenir :

```text
mu_A
mu_B
```

Puis plusieurs points intermédiaires sont calculés :

```text
z(t) = (1 - t) × mu_A + t × mu_B
```

avec :

```text
t allant de 0 à 1
```

Pour le CVAE, le même label de classe est conservé pendant toute l'interpolation.

Les expériences ont notamment été réalisées sur les classes :

```text
0 -> T-shirt/top
1 -> Trouser
5 -> Sandal
9 -> Ankle boot
```

Les transitions obtenues sont globalement progressives, ce qui indique une certaine continuité locale de l'espace latent.

Cette analyse reste qualitative : une interpolation fluide entre quelques paires d'images ne démontre pas à elle seule que l'ensemble de l'espace latent est parfaitement organisé.

Les figures et les informations sur les paires utilisées sont disponibles dans :

```text
results/interpolations/
```

---

## 11. Structure du sous-projet

```text
david_fashion_mnist/
|
+-- models/
|   +-- vae.py
|   +-- cvae.py
|   +-- __init__.py
|
+-- training/
|   +-- losses.py
|   +-- train_vae.py
|   +-- train_cvae.py
|   +-- __init__.py
|
+-- evaluation/
|   +-- evaluate.py
|   +-- ablation_beta.py
|   +-- latent_visualization.py
|   +-- interpolation.py
|   +-- __init__.py
|
+-- results/
|   +-- evaluation_metrics.csv
|   +-- training_histories/
|   +-- reconstructions/
|   +-- generations/
|   +-- ablation/
|   +-- latent_spaces/
|   +-- interpolations/
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

Les données Fashion-MNIST et les checkpoints PyTorch ne sont pas versionnés dans Git.

---

## 12. Installation

Depuis le dossier :

```text
projects/david_fashion_mnist
```

créer un environnement virtuel :

```powershell
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Le dataset Fashion-MNIST est téléchargé automatiquement par `torchvision` lors de la première exécution.

---

## 13. Entraînement

Les commandes doivent être exécutées depuis :

```text
projects/david_fashion_mnist
```

### VAE beta = 0.1

```powershell
python -m training.train_vae --beta 0.1 --epochs 20
```

### VAE beta = 1

```powershell
python -m training.train_vae --beta 1 --epochs 20
```

### VAE beta = 4

```powershell
python -m training.train_vae --beta 4 --epochs 20
```

### CVAE beta = 0.1

```powershell
python -m training.train_cvae --beta 0.1 --epochs 20
```

### CVAE beta = 1

```powershell
python -m training.train_cvae --beta 1 --epochs 20
```

### CVAE beta = 4

```powershell
python -m training.train_cvae --beta 4 --epochs 20
```

---

## 14. Évaluation

Après entraînement et présence des checkpoints locaux :

```powershell
python -m evaluation.evaluate
```

Le script :

1. charge les modèles entraînés ;
2. utilise les 10 000 images du jeu officiel de test ;
3. calcule les pertes de reconstruction et KL ;
4. sauvegarde les métriques ;
5. produit les grilles de reconstruction ;
6. produit les générations.

---

## 15. Étude d'ablation

```powershell
python -m evaluation.ablation_beta
```

Cette analyse compare les trois valeurs de `beta` pour le VAE et le CVAE.

---

## 16. Visualisation de l'espace latent

```powershell
python -m evaluation.latent_visualization
```

Le script produit des projections :

- PCA ;
- UMAP.

---

## 17. Interpolation dans l'espace latent

```powershell
python -m evaluation.interpolation
```

Le script encode deux images d'une même classe puis génère plusieurs points intermédiaires dans l'espace latent.

---

## 18. Difficultés rencontrées et solutions

### Compromis reconstruction / régularisation

Une faible valeur de `beta` améliore la reconstruction mais régularise moins fortement l'espace latent.

À l'inverse, une valeur élevée réduit le KL mais dégrade progressivement la précision des reconstructions.

L'étude d'ablation permet de mettre quantitativement et qualitativement ce compromis en évidence.

### Reconstruction et génération ne mesurent pas exactement la même chose

Un modèle capable de bien reconstruire les images observées n'est pas nécessairement capable de produire de bonnes images en échantillonnant directement :

```text
z ~ N(0, I)
```

Le CVAE avec `beta = 0.1` illustre particulièrement ce phénomène : sa reconstruction est bonne mais certaines générations conditionnelles restent instables.

### Interprétation des visualisations latentes

PCA et UMAP projettent un espace latent de dimension 16 vers seulement deux dimensions.

Les figures permettent d'étudier certaines structures, mais elles ne constituent pas une représentation complète de l'espace latent.

### Reproductibilité

Une graine fixe `42` est utilisée afin de stabiliser :

- la séparation train/validation ;
- l'initialisation ;
- l'ordre des données ;
- la sélection des échantillons utilisés pour certaines analyses.

---

## 19. Limites de l'expérience actuelle

Tous les meilleurs checkpoints des six expériences ont été obtenus à l'époque 20, qui correspond également à la dernière époque programmée.

Cela indique que les modèles pouvaient potentiellement continuer à progresser au-delà de 20 époques.

Les expériences actuelles constituent donc une étude comparative cohérente avec un budget fixe de 20 époques, mais pas nécessairement l'entraînement définitif optimal de chaque modèle.

Une prochaine phase utilisera :

- un nombre maximal d'époques plus élevé ;
- l'early stopping ;
- le suivi des expériences avec MLflow.

---

## 20. Prochaine étape : MLflow

Le suivi expérimental actuel repose sur :

- les checkpoints PyTorch ;
- les historiques CSV ;
- les graphiques ;
- les métriques d'évaluation.

La prochaine version intégrera **MLflow** afin de centraliser :

- les hyperparamètres ;
- les métriques par époque ;
- la loss de reconstruction ;
- le terme KL ;
- la loss de validation ;
- la meilleure époque ;
- les checkpoints ;
- les figures et autres artefacts ;
- les informations permettant de comparer les différentes expériences.

L'intégration MLflow sera accompagnée d'un mécanisme d'early stopping pour les futurs entraînements.

---

## 21. Conclusion

Les expériences réalisées sur Fashion-MNIST montrent clairement le compromis entre qualité de reconstruction et régularisation de l'espace latent.

Une faible valeur de `beta` favorise la reconstruction mais produit un espace latent moins contraint.

Une forte valeur de `beta` rapproche davantage la distribution latente du prior mais entraîne une perte de détails et de diversité.

Parmi les configurations étudiées, `beta = 1` constitue le compromis qualitatif le plus équilibré, particulièrement pour le CVAE.

Le CVAE apporte surtout un avantage dans la génération contrôlée par classe, tandis que les performances de reconstruction du VAE et du CVAE restent relativement proches pour une même valeur de `beta`.

Les expériences d'ablation, les visualisations PCA/UMAP et les interpolations permettent de compléter les métriques quantitatives par une analyse qualitative de la structure apprise par les modèles.