# PFE Fashion-MNIST — VAE / CVAE
## Résumé complet du projet, concepts, architecture, protocole, fichiers, résultats et préparation à la soutenance

> **Objectif de ce document**  
> Ce fichier est une synthèse unique de tout le projet, depuis la compréhension du sujet jusqu'à la livraison des modèles pour l'application web. Il sert de document de révision pour la soutenance.

---

# 1. Sujet et objectif général

Le projet consiste à construire, entraîner et évaluer deux modèles génératifs sur **Fashion-MNIST** :

- un **VAE** (*Variational Autoencoder*) ;
- un **CVAE** (*Conditional Variational Autoencoder*).

Les objectifs étaient :

1. apprendre un espace latent continu capable de reconstruire des images Fashion-MNIST ;
2. générer de nouvelles images ;
3. avec le CVAE, contrôler explicitement la classe générée ;
4. rendre le protocole reproductible ;
5. suivre les expériences avec MLflow ;
6. comparer plusieurs valeurs de `beta` ;
7. vérifier la robustesse sur plusieurs seeds ;
8. mesurer la cohérence conditionnelle et la diversité ;
9. évaluer finalement les modèles sur le test officiel ;
10. livrer les checkpoints finaux pour l'application web.

Le projet ne s'est donc pas limité à « entraîner un réseau ». Il a couvert toute la chaîne expérimentale :

```text
théorie
→ données
→ architecture
→ entraînement
→ choix des hyperparamètres
→ reproductibilité
→ suivi MLflow
→ évaluation
→ analyse des résultats
→ livraison des modèles
```

---

# 2. Fashion-MNIST

Fashion-MNIST contient des images de vêtements en niveaux de gris.

Chaque image est :

```text
1 x 28 x 28
```

Les 10 classes sont :

| ID | Classe |
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

Le dataset officiel contient :

- 60 000 images d'entraînement ;
- 10 000 images de test.

Dans le protocole final, les 60 000 images officielles d'entraînement sont séparées en :

```text
54 000 train
6 000 validation
```

avec :

```text
split_seed = 42
```

Le test officiel de 10 000 images est utilisé à la fin pour mesurer la généralisation.

Le preprocessing final utilise uniquement :

```python
ToTensor()
```

donc les pixels sont dans `[0,1]`.

---

# 3. Autoencodeur, VAE et espace latent

## 3.1 Autoencodeur classique

Un autoencodeur contient :

```text
image x
  ↓
encodeur
  ↓
représentation latente z
  ↓
décodeur
  ↓
reconstruction x_hat
```

L'objectif est de reconstruire l'entrée après compression.

La limite d'un autoencodeur classique est que son espace latent n'est pas forcément suffisamment régulier pour pouvoir échantillonner facilement de nouveaux points et générer des images réalistes.

---

# 4. VAE — Variational Autoencoder

Le VAE apprend une **distribution** latente au lieu d'un seul vecteur déterministe.

L'encodeur produit :

```text
mu
logvar
```

La distribution approximative est :

```text
q(z|x) = N(mu, sigma²)
```

avec :

```text
sigma = exp(0.5 * logvar)
```

On échantillonne :

```text
epsilon ~ N(0,I)
z = mu + sigma * epsilon
```

Cette écriture est le **reparameterization trick**.

## Pourquoi le reparameterization trick ?

Un échantillonnage direct bloquerait la rétropropagation. En écrivant le bruit comme une variable externe `epsilon`, le réseau reste différentiable par rapport à `mu` et `logvar`.

---

# 5. Fonction de perte du VAE

Nous utilisons :

```text
Loss = Reconstruction BCE + beta * KL
```

## Reconstruction BCE

La Binary Cross Entropy mesure la différence entre l'image originale et la reconstruction.

Plus la BCE est faible, meilleure est la reconstruction.

## Divergence KL

Le terme KL force la distribution latente à rester proche du prior :

```text
N(0,I)
```

Formule :

```text
KL = -0.5 * Σ(1 + logvar - mu² - exp(logvar))
```

Le KL sert donc à régulariser l'espace latent et à rendre possible la génération depuis un simple tirage `z ~ N(0,I)`.

---

# 6. Rôle de beta

Nous avons testé :

```text
beta ∈ {0.1, 1, 4}
```

`beta` contrôle le compromis entre reconstruction et régularisation.

### beta = 0.1
- reconstruction très bonne ;
- KL élevé ;
- latent moins régularisé ;
- génération plus instable.

### beta = 1
- compromis équilibré ;
- bonne reconstruction ;
- latent suffisamment régularisé ;
- bon comportement génératif.

### beta = 4
- forte régularisation ;
- KL faible ;
- reconstructions plus dégradées ;
- générations plus prototypiques ;
- diversité plus faible.

## Point méthodologique crucial

On ne compare pas directement les pertes totales entre différents beta, car l'objectif change :

```text
Loss_beta0.1 = BCE + 0.1 * KL
Loss_beta4   = BCE + 4 * KL
```

La sélection de beta doit donc s'appuyer sur plusieurs critères : reconstruction, KL, cohérence, diversité et inspection qualitative.

---

# 7. CVAE — Conditional Variational Autoencoder

Le CVAE ajoute une condition `y`, ici la classe Fashion-MNIST.

Schéma :

```text
image x + classe y
        ↓
      encodeur
        ↓
     mu, logvar
        ↓
         z
        ↓
    z + classe y
        ↓
      décodeur
        ↓
       image
```

Avec le CVAE, on peut demander :

```text
"Génère une Sneaker"
```

La classe fixe le type d'objet, tandis que `z` contrôle les variations intra-classe.

La fonction particulièrement importante est :

```python
cvae.decode(z, labels)
```

Elle permet d'imposer la classe et de réutiliser exactement les mêmes vecteurs latents pour des comparaisons contrôlées.

---

# 8. Architecture finale

Configuration principale :

```text
latent_dim = 16
hidden_dim = 256
image = 1 x 28 x 28
```

## VAE

Encodeur convolutionnel :

```text
1 canal
→ Conv 32
→ Conv 64
→ fully-connected
→ mu / logvar
```

Décodeur :

```text
z
→ fully-connected
→ convolutions transposées
→ Sigmoid
→ image 1 x 28 x 28
```

## CVAE

Même principe, mais les labels sont injectés dans l'encodeur et le décodeur sous forme conditionnelle.

---

# 9. Protocole de reproductibilité

Nous avons séparé deux seeds :

```text
training_seed
split_seed
```

## split_seed

Contrôle uniquement le split train/validation.

Valeur finale :

```text
42
```

## training_seed

Contrôle notamment :

- initialisation du modèle ;
- ordre des batches ;
- opérations aléatoires de l'entraînement.

Seeds finales :

```text
0
42
123
```

Nous avons vérifié que le split train/validation reste strictement identique pour les trois training seeds.

Cela garantit :

```text
mêmes données
+ initialisations différentes
= comparaison multi-seed correcte
```

---

# 10. Entraînement

Configuration principale :

```text
optimizer      = Adam
learning_rate  = 0.001
batch_size     = 128
latent_dim     = 16
hidden_dim     = 256
max_epochs     = 100
patience       = 10
min_delta      = 0
split_seed     = 42
```

L'early stopping surveille :

```text
validation_total
```

Le checkpoint conservé est celui correspondant au meilleur epoch de validation, et non nécessairement au dernier epoch exécuté.

---

# 11. MLflow

MLflow a été intégré pour suivre les expériences.

Backend :

```text
mlflow.db
```

Artefacts :

```text
mlartifacts/
```

MLflow conserve notamment :

- modèle ;
- beta ;
- seeds ;
- hyperparamètres ;
- losses ;
- meilleur epoch ;
- durée ;
- métriques d'évaluation ;
- artefacts CSV et images.

Une sauvegarde persistante a aussi été réalisée sur Google Drive.

---

# 12. Environnement

## Local
- Windows ;
- VS Code ;
- PowerShell ;
- environnement virtuel Python.

Projet équipe :

```text
D:\vae-cvae-image-generation-equipe
```

Sous-projet :

```text
projects\david_fashion_mnist
```

## Colab
Les entraînements finaux multi-seed ont été réalisés sur :

```text
GPU = Tesla T4
PyTorch = 2.11.0+cu128
CUDA = True
```

Les runs CPU exploratoires/interrompus ont été conservés comme traces mais exclus des statistiques finales.


---

# 13. Screening de beta

Nous avons étudié `beta = 0.1`, `1` et `4` pour le VAE et le CVAE.

## Résultats principaux seed 42

| Modèle | beta | Best epoch | Val total | Recon BCE | KL |
|---|---:|---:|---:|---:|---:|
| VAE | 0.1 | 98 | 214.371687 | 210.257216 | 41.144712 |
| VAE | 1 | 83 | 235.645208 | 221.380782 | 14.264426 |
| VAE | 4 | 50 | 265.171900 | 237.922530 | 6.812342 |
| CVAE | 0.1 | 58 | 214.321668 | 210.455290 | 38.663779 |
| CVAE | 1 | 92 | 233.197299 | 221.200660 | 11.996638 |
| CVAE | 4 | 94 | 255.656524 | 236.919639 | 4.684221 |

Interprétation :

- `beta=0.1` : reconstruction excellente mais latent peu régularisé ;
- `beta=4` : régularisation forte, reconstruction dégradée et générations plus prototypiques ;
- `beta=1` : meilleur compromis global.

---

# 14. Classifieur indépendant Fashion-MNIST

Pour mesurer objectivement si les images générées correspondent aux classes demandées, nous avons utilisé un classifieur indépendant.

Accuracy validation :

```text
92.6167 %
```

Performances approximatives par classe :

| Classe | Accuracy |
|---|---:|
| T-shirt/top | 88.30 % |
| Trouser | 99.51 % |
| Pullover | 93.14 % |
| Dress | 93.77 % |
| Coat | 88.30 % |
| Sandal | 96.90 % |
| Shirt | 74.67 % |
| Sneaker | 94.60 % |
| Bag | 99.33 % |
| Ankle boot | 98.17 % |

Ce classifieur n'entraîne pas le CVAE. Il sert d'**évaluateur externe**.

---

# 15. Cohérence conditionnelle

Pour chaque image générée :

1. on choisit une classe demandée `y` ;
2. le CVAE génère une image conditionnée par `y` ;
3. le classifieur indépendant prédit la classe ;
4. on vérifie si la prédiction correspond à `y`.

Métrique :

```text
conditional_accuracy
=
images reconnues comme la classe demandée
/
nombre total d'images générées
```

---

# 16. Banque latente contrôlée

Pour éviter que les différences viennent de tirages aléatoires différents de `z`, nous avons généré une banque unique :

```text
z ~ N(0,I)
```

puis utilisé exactement les mêmes `z` :

- entre modèles ;
- entre seeds ;
- entre classes.

Cela rend les comparaisons beaucoup plus propres.

---

# 17. Screening de cohérence selon beta

Résultats seed 42 :

| beta | Conditional accuracy |
|---:|---:|
| 0.1 | 58.38 % |
| 1 | 83.38 % |
| 4 | 84.10 % |

`beta=4` n'améliore la cohérence que de :

```text
84.10 - 83.38 = 0.72 point
```

par rapport à `beta=1`.

Ce gain est faible et doit être comparé à la baisse de diversité et à la dégradation de reconstruction observées avec `beta=4`.

---

# 18. Diversité des générations

Deux familles de métriques ont été utilisées.

## 18.1 Diversité pixel

Distance RMS moyenne paire-à-paire entre les images d'une même classe.

Elle mesure les différences visuelles directement dans l'espace des pixels.

## 18.2 Diversité sémantique

On extrait les représentations pénultièmes du classifieur indépendant, puis on calcule la distance cosinus moyenne entre toutes les paires.

Cette mesure reflète des différences dans un espace de représentation plus sémantique.

---

# 19. ALL vs COHERENT

Les métriques de diversité sont calculées :

```text
ALL
```

sur toutes les générations ;

et :

```text
COHERENT
```

uniquement sur les générations reconnues dans la classe demandée.

La version `COHERENT` est particulièrement importante : un modèle ne doit pas être récompensé pour produire des images très différentes simplement parce qu'elles sont hors classe.

---

# 20. Screening diversité beta

Pour seed 42 :

| beta | Cohérence | Pixel ALL | Pixel cohérent | Feature ALL | Feature cohérent |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 58.38 % | 0.290020 | 0.272232 | 0.397883 | 0.270561 |
| 1 | 83.38 % | 0.262197 | 0.256501 | 0.190108 | 0.161098 |
| 4 | 84.10 % | 0.242861 | 0.238357 | 0.160300 | 0.139451 |

Par rapport à `beta=4`, `beta=1` conserve presque la même cohérence tout en offrant environ :

```text
+7.6 % de diversité pixel cohérente
+15.5 % de diversité sémantique cohérente
```

C'est une justification forte du choix final :

```text
beta = 1
```

---

# 21. Entraînements finaux multi-seed

Les seeds finales sont :

```text
0
42
123
```

Tous les runs statistiques finaux utilisent :

```text
beta = 1
split_seed = 42
GPU = Tesla T4
```

## CVAE

| Seed | Best epoch | Best val loss | Epochs exécutés | Early stopping |
|---:|---:|---:|---:|---|
| 0 | 76 | 233.441762 | 86 | Oui |
| 42 | 92 | 233.197299 | 100 | Non |
| 123 | 83 | 233.991843 | 93 | Oui |

Résumé :

```text
233.543634 ± 0.406950
```

## VAE

| Seed | Best epoch | Best val loss | Epochs exécutés | Early stopping |
|---:|---:|---:|---:|---|
| 0 | 64 | 235.884890 | 74 | Oui |
| 42 | 83 | 235.645208 | 93 | Oui |
| 123 | 74 | 235.694453 | 84 | Oui |

Résumé :

```text
235.741517 ± 0.126583
```

À `beta=1`, la comparaison de la loss totale VAE/CVAE est plus défendable car les deux utilisent la même pondération du KL.

---

# 22. Correction méthodologique : reconstruction déterministe

Une première version de `evaluate.py` utilisait le forward complet du modèle pour calculer MSE et SSIM.

Or le forward échantillonne :

```text
z ~ q(z|x)
```

Cela ajoute du bruit aux métriques de reconstruction.

Nous avons corrigé l'évaluation finale.

## Métriques sampled

On garde un échantillonnage latent pour :

```text
evaluation_total_sampled
evaluation_reconstruction_bce_sampled
evaluation_kl
```

## Métriques déterministes

Pour la reconstruction finale :

```text
z = mu
reconstruction = decode(mu)
```

Puis on calcule :

```text
evaluation_reconstruction_bce_deterministic
evaluation_mse_deterministic
evaluation_ssim_deterministic
```

Cette distinction permet de conserver à la fois :

- la nature probabiliste du VAE/CVAE ;
- une comparaison stable de la fidélité de reconstruction.

---

# 23. Validation finale multi-seed

Évaluation sur les 6 000 images de validation.

| Modèle | Total sampled | BCE sampled | KL | BCE déterministe | MSE déterministe | SSIM déterministe |
|---|---:|---:|---:|---:|---:|---:|
| CVAE | 233.624677 ± 0.375595 | 221.880682 ± 0.806827 | 11.743995 ± 0.442963 | 218.407079 ± 0.963041 | 0.012535 ± 0.000439 | 0.685189 ± 0.005675 |
| VAE | 235.834803 ± 0.183033 | 221.195649 ± 0.384356 | 14.639155 ± 0.353505 | 217.289731 ± 0.499222 | 0.012058 ± 0.000214 | 0.688722 ± 0.001217 |

Interprétation :

- le CVAE possède un total plus faible, principalement grâce à un KL plus faible ;
- le VAE obtient une reconstruction déterministe légèrement meilleure.

Il n'y a pas de contradiction : le CVAE est surtout conçu pour ajouter le contrôle conditionnel.

---

# 24. Cohérence conditionnelle finale multi-seed

Pour chaque seed :

```text
10 classes x 1000 images = 10 000 générations
```

Sur trois seeds :

```text
30 000 générations
```

Résultats :

| Seed | Conditional accuracy |
|---:|---:|
| 0 | 81.69 % |
| 42 | 83.38 % |
| 123 | 82.19 % |

Résumé :

```text
82.42 % ± 0.87 point de pourcentage
```

Total :

```text
24 726 / 30 000 générations cohérentes
```

---

# 25. Cohérence par classe

| Classe | Moyenne | Écart-type |
|---|---:|---:|
| T-shirt/top | 83.47 % | 0.38 pp |
| Trouser | 98.87 % | 0.21 pp |
| Pullover | 90.57 % | 3.96 pp |
| Dress | 88.97 % | 0.31 pp |
| Coat | 34.70 % | 3.33 pp |
| Sandal | 72.23 % | 0.45 pp |
| Shirt | 72.40 % | 6.32 pp |
| Sneaker | 87.00 % | 4.00 pp |
| Bag | 96.97 % | 0.96 pp |
| Ankle boot | 99.03 % | 0.57 pp |

Classes les plus faciles :

```text
Ankle boot
Trouser
Bag
```

Classe la plus difficile :

```text
Coat
```

Les confusions entre `Coat`, `Shirt`, `Pullover` et `T-shirt/top` sont plausibles car ces catégories sont visuellement proches.

---

# 26. Diversité finale multi-seed

| Seed | Cond. acc. | Pixel ALL | Pixel cohérent | Feature ALL | Feature cohérent |
|---:|---:|---:|---:|---:|---:|
| 0 | 81.69 % | 0.262735 | 0.256576 | 0.202977 | 0.168043 |
| 42 | 83.38 % | 0.262197 | 0.256501 | 0.190108 | 0.161098 |
| 123 | 82.19 % | 0.263597 | 0.257258 | 0.197235 | 0.164112 |

Résumé :

```text
Conditional accuracy
82.42 % ± 0.87 pp

Pixel diversity ALL
0.262843 ± 0.000706

Pixel diversity COHERENT
0.256778 ± 0.000417

Semantic diversity ALL
0.196773 ± 0.006447

Semantic diversity COHERENT
0.164418 ± 0.003483
```

La diversité pixel cohérente est particulièrement stable entre seeds, ce qui renforce la robustesse descriptive du comportement génératif.



---

# 27. Test officiel final

Après gel définitif des choix :

- architecture ;
- beta ;
- seeds ;
- checkpoints ;
- métriques ;
- protocole ;

les 6 checkpoints ont été évalués sur les 10 000 images officielles du test.

Cela représente :

```text
6 checkpoints x 10 000 images
= 60 000 évaluations
```

## Résultats individuels

### CVAE

| Seed | Total sampled | BCE sampled | KL | BCE det. | MSE det. | SSIM det. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 234.268327 | 222.330980 | 11.937346 | 218.813831 | 0.012245 | 0.685737 |
| 42 | 234.085058 | 222.158338 | 11.926720 | 218.521637 | 0.012132 | 0.689693 |
| 123 | 234.838936 | 223.659181 | 11.179755 | 220.338870 | 0.012960 | 0.678135 |

### VAE

| Seed | Total sampled | BCE sampled | KL | BCE det. | MSE det. | SSIM det. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 236.830111 | 222.240038 | 14.590073 | 217.963688 | 0.011873 | 0.687844 |
| 42 | 236.451482 | 222.270173 | 14.181309 | 218.630514 | 0.012194 | 0.686760 |
| 123 | 236.493662 | 221.595701 | 14.897960 | 217.777641 | 0.011838 | 0.688881 |

## Résumé test officiel

| Modèle | Total sampled ↓ | KL ↓ | BCE déterministe ↓ | MSE déterministe ↓ | SSIM déterministe ↑ |
|---|---:|---:|---:|---:|---:|
| CVAE beta=1 | **234.397 ± 0.393** | **11.681 ± 0.434** | 219.225 ± 0.976 | 0.012446 ± 0.000449 | 0.684522 ± 0.005874 |
| VAE beta=1 | 236.592 ± 0.208 | 14.556 ± 0.360 | **218.124 ± 0.448** | **0.011968 ± 0.000196** | **0.687828 ± 0.001061** |

Conclusion :

```text
VAE  → meilleure fidélité de reconstruction
CVAE → génération conditionnelle contrôlée
```

Le CVAE possède un objectif total plus faible à `beta=1` principalement grâce à un KL plus faible, tandis que le VAE reconstruit légèrement mieux selon BCE, MSE et SSIM déterministes.

---

# 28. Transparence concernant le test officiel

Le test officiel avait été consulté lors d'une phase exploratoire préliminaire du projet.

Il faut donc présenter honnêtement le protocole :

> Le test officiel a été utilisé de manière exploratoire pendant une phase préliminaire. Ensuite, les décisions finales ont été prises sur la validation. Une dernière évaluation sur le test officiel a été réalisée après gel définitif des choix, sans nouvelle sélection après consultation des résultats.

Il ne faut pas dire que le test n'avait jamais été observé auparavant.

---

# 29. Pourquoi trois seeds ?

Une seule seed peut produire un résultat favorable ou défavorable par hasard.

Nous avons donc utilisé :

```text
0, 42, 123
```

puis calculé :

```text
moyenne ± écart-type
```

Cela montre la stabilité descriptive des conclusions.

Avec seulement `n=3`, il ne faut toutefois pas parler de forte significativité statistique. La formulation correcte est :

```text
stabilité descriptive sur trois initialisations
```

---

# 30. Structure des fichiers principaux

Les fichiers principaux créés, modifiés ou utilisés dans le sous-projet sont :

```text
projects/
└── david_fashion_mnist/
    ├── models/
    │   ├── vae.py
    │   ├── cvae.py
    │   └── fashion_classifier.py
    │
    ├── training/
    │   └── train_vae.py
    │
    ├── evaluation/
    │   ├── evaluate.py
    │   ├── evaluate_classifier.py
    │   ├── conditional_coherence.py
    │   └── generation_diversity.py
    │
    ├── checkpoints/
    │   └── *.pt
    │
    ├── results/
    │   └── report_assets/
    │
    ├── mlflow.db
    └── mlartifacts/
```

Cette arborescence présente les composants les plus importants du travail ; le dépôt peut contenir d'autres fichiers auxiliaires.

---

# 31. `models/vae.py`

Rôle :

```text
définition du Variational Autoencoder
```

Ce fichier contient notamment :

- l'encodeur convolutionnel ;
- les couches produisant `mu` et `logvar` ;
- le reparameterization trick ;
- le décodeur ;
- le forward ;
- les fonctions nécessaires à la génération.

Entrée :

```text
[N, 1, 28, 28]
```

Le latent final est de dimension :

```text
16
```

---

# 32. `models/cvae.py`

Rôle :

```text
définition du Conditional Variational Autoencoder
```

La différence fondamentale est l'ajout du label Fashion-MNIST comme condition.

Fonction importante :

```python
decode(z, labels)
```

Elle est utilisée pour :

- imposer une classe ;
- générer de manière contrôlée ;
- réutiliser le même `z` entre différents modèles ;
- effectuer les évaluations de cohérence et diversité ;
- alimenter l'application web.

---

# 33. `models/fashion_classifier.py`

Rôle :

```text
classifieur indépendant Fashion-MNIST
```

Il sert à :

- évaluer la cohérence des générations ;
- produire des prédictions de classe ;
- extraire une représentation pénultième de dimension 128 pour la diversité sémantique.

Le classifieur n'est pas utilisé pour entraîner le CVAE.

---

# 34. `training/train_vae.py`

Fichier central de l'entraînement.

Il gère notamment :

- sélection CPU/GPU ;
- random seeds ;
- DataLoaders ;
- split train/validation ;
- entraînement VAE ;
- entraînement CVAE ;
- calcul de la loss ;
- validation ;
- early stopping ;
- checkpoints ;
- MLflow ;
- configuration des runs.

Amélioration méthodologique importante :

```text
training_seed séparé de split_seed
```

Cela permet de changer l'initialisation sans changer les données de validation.

---

# 35. `evaluation/evaluate.py`

Rôle :

```text
évaluation quantitative principale VAE/CVAE
```

Il accepte notamment :

```text
--split validation
--split test
```

Métriques principales :

```text
evaluation_total_sampled
evaluation_reconstruction_bce_sampled
evaluation_kl
evaluation_reconstruction_bce_deterministic
evaluation_mse_deterministic
evaluation_ssim_deterministic
```

Il produit également des grilles de reconstruction, des générations et un CSV de résultats.

Correction importante :

```text
MSE / SSIM / BCE déterministe utilisent decode(mu)
```

---

# 36. `evaluation/evaluate_classifier.py`

Rôle :

- charger le classifieur indépendant ;
- calculer ses métriques ;
- normaliser les matrices de confusion ;
- sauvegarder les matrices CSV ;
- fournir des fonctions réutilisées par les autres scripts d'évaluation.

---

# 37. `evaluation/conditional_coherence.py`

Rôle :

```text
mesurer si le CVAE respecte la classe demandée
```

Version finale multi-seed :

```text
beta = 1
seeds = 0,42,123
1000 images par classe
10 classes
même banque latente
```

Sorties principales :

```text
conditional_coherence_multiseed_runs.csv
conditional_coherence_multiseed_summary.csv
conditional_coherence_multiseed_per_class.csv
conditional_coherence_multiseed_per_class_summary.csv
```

Il produit aussi :

- matrices de confusion ;
- matrices normalisées ;
- figures ;
- grilles de générations contrôlées.

---

# 38. `evaluation/generation_diversity.py`

Rôle :

```text
mesurer la diversité intra-classe des générations CVAE
```

Métriques :

```text
pixel_pairwise_rms
feature_cosine_diversity
```

Versions :

```text
ALL
COHERENT
```

La version finale :

- compare les trois training seeds ;
- utilise une banque `z` commune ;
- vérifie beta=1 ;
- vérifie les seeds ;
- agrège moyenne ± écart-type ;
- journalise les résultats dans MLflow.

Sorties principales :

```text
generation_diversity_multiseed_runs.csv
generation_diversity_multiseed_summary.csv
generation_diversity_multiseed_per_class.csv
generation_diversity_multiseed_per_class_summary.csv
```

---

# 39. `checkpoints/`

Ce dossier contient les poids PyTorch :

```text
*.pt
```

Ils sont ignorés par Git pour éviter de stocker de gros fichiers binaires dans l'historique.

Checkpoints finaux scientifiques :

```text
VAE
vae_beta_1_seed0_gpu_multiseed.pt
vae_beta_1_seed42_final.pt
vae_beta_1_seed123_gpu_multiseed.pt

CVAE
cvae_beta_1_seed0_gpu_multiseed.pt
cvae_beta_1_seed42_final.pt
cvae_beta_1_seed123_gpu_multiseed.pt
```

Ils sont sauvegardés de manière persistante sur Google Drive.

---

# 40. Modèles destinés à l'application web

Pour l'application, un seul checkpoint représentatif par architecture est nécessaire.

Nous avons retenu :

```text
vae_beta_1_seed42_final.pt
cvae_beta_1_seed42_final.pt
```

Pourquoi seed 42 ?

Parce que ces checkpoints ont la meilleure perte validation parmi les trois seeds finales.

## VAE

```text
seed 0   : 235.884890
seed 42  : 235.645208  ← meilleur
seed 123 : 235.694453
```

## CVAE

```text
seed 0   : 233.441762
seed 42  : 233.197299  ← meilleur
seed 123 : 233.991843
```

---

# 41. Package de livraison application

Archive créée :

```text
fashion_mnist_app_models.zip
```

Contenu :

```text
fashion_mnist_app_models/
├── vae_beta_1_seed42_final.pt
├── cvae_beta_1_seed42_final.pt
├── model_manifest.json
└── README_MODELS.txt
```

Copie locale :

```text
D:\PFE_Fashion_MNIST_delivery\
```

Le ZIP peut être envoyé directement au collègue chargé de l'application web.

GitHub n'est pas obligatoire pour distribuer les poids.

---

# 42. SHA256 des modèles application

VAE :

```text
AA480900FEF0CDF2C8C5DBA86EE9C3481E511FAB0193F7FA2EFD88E637C623EC
```

CVAE :

```text
12F89DD426301E19CDBD87F6A4DC37EEF6CCD016DFBDA5DCA6A43ABC06FA0F87
```

Ces empreintes permettent de vérifier que le collègue utilise exactement les bons checkpoints.

---

# 43. `model_manifest.json`

Ce fichier contient les métadonnées du package :

- nom du modèle ;
- nom du fichier ;
- taille ;
- beta ;
- training seed ;
- split seed ;
- latent dimension ;
- hidden dimension ;
- SHA256.

Il agit comme un mini contrat de livraison.

---

# 44. `README_MODELS.txt`

Il explique au collègue :

- quels checkpoints utiliser ;
- leur configuration ;
- les classes Fashion-MNIST ;
- où se trouvent `models/vae.py` et `models/cvae.py` ;
- comment interpréter les modèles.

---

# 45. `results/report_assets/`

Ce dossier a été préparé pour le mémoire et la soutenance.

Structure :

```text
report_assets/
├── tables/
│   ├── csv/
│   ├── xlsx/
│   └── png/
├── figures/
│   ├── png/
│   ├── svg/
│   └── pdf/
└── presentation/
    └── ready_to_insert/
```

Principaux tableaux :

```text
table_final_official_test.csv
table_final_validation.csv
table_final_cvae_generation.csv
table_validation_vs_test.csv
fashion_mnist_final_results.xlsx
```

Principales figures :

```text
fig_final_test_mse
fig_final_test_ssim
fig_cvae_conditional_coherence_by_seed
fig_cvae_conditional_coherence_per_class
fig_cvae_pixel_diversity_by_seed
fig_cvae_semantic_diversity_by_seed
fig_cvae_pixel_diversity_per_class
fig_cvae_semantic_diversity_per_class
```

Le dossier `presentation/ready_to_insert` contient les éléments les plus utiles à insérer directement dans les slides.



---

# 46. Git et reproductibilité

Deux dépôts ont été maintenus.

## Dépôt équipe

Branche :

```text
feature/david-fashion-mnist-mlflow
```

Commits finaux importants :

```text
0286462 chore: clean generated SVG whitespace
61106df docs: add final Fashion-MNIST report assets
13dd311 feat: add final multiseed CVAE diversity evaluation
a51bcac feat: add final multiseed CVAE coherence evaluation
563c628 fix: make final reconstruction evaluation deterministic
```

## Dépôt personnel

Branche :

```text
main
```

Commits finaux importants :

```text
ff25f6a docs: add final Fashion-MNIST report assets
93fbcb2 feat: add final multiseed CVAE diversity evaluation
c3e8d89 feat: add final multiseed CVAE coherence evaluation
7188c7c fix: make final reconstruction evaluation deterministic
```

Les deux dépôts ont été poussés et vérifiés propres.

---

# 47. `.gitignore`

Les éléments lourds ou temporaires sont exclus :

- environnement virtuel ;
- caches Python ;
- datasets locaux ;
- checkpoints `.pt` ;
- base MLflow locale ;
- artefacts MLflow ;
- fichiers temporaires.

Les figures, tableaux et résultats destinés au rapport sont versionnés.

---

# 48. Comment l'application utilise le VAE

Le VAE peut servir à :

```text
image
→ encodeur
→ mu / logvar
→ z
→ décodeur
→ reconstruction
```

ou à générer librement depuis un latent aléatoire :

```text
z ~ N(0,I)
→ decoder
→ image
```

---

# 49. Comment l'application utilise le CVAE

L'utilisateur choisit une classe, par exemple :

```text
Sneaker
```

Le backend associe :

```text
label = 7
```

puis :

```text
z ~ N(0,I)
image = cvae.decode(z, label)
```

Le CVAE est donc particulièrement adapté à une démonstration web où l'utilisateur choisit la catégorie à générer.

---

# 50. Le classifieur est-il nécessaire dans l'application ?

Non.

Le classifieur a surtout servi à l'évaluation scientifique.

L'application peut fonctionner uniquement avec :

```text
VAE
CVAE
```

Le classifieur ne devient utile que si l'équipe souhaite afficher une prédiction automatique sur l'image générée.

---

# 51. Conclusion scientifique globale

## VAE

Avantages :

- reconstruction légèrement meilleure ;
- architecture plus simple ;
- bon modèle génératif non conditionnel.

Résultats test :

```text
MSE  = 0.011968 ± 0.000196
SSIM = 0.687828 ± 0.001061
BCE  = 218.124 ± 0.448
```

## CVAE

Avantages :

- génération contrôlée par classe ;
- cohérence conditionnelle stable ;
- diversité cohérente stable ;
- KL plus faible.

Résultats génération :

```text
cohérence conditionnelle
82.42 % ± 0.87 pp

diversité pixel cohérente
0.256778 ± 0.000417

diversité sémantique cohérente
0.164418 ± 0.003483
```

## Message principal

```text
VAE  → meilleur pour la fidélité de reconstruction
CVAE → meilleur lorsque l'on veut contrôler la classe générée
```

Il ne faut donc pas dire qu'un modèle est « meilleur » de manière absolue. Ils répondent à des objectifs légèrement différents.

---

# 52. Limites du projet

Principales limites :

1. Fashion-MNIST est un dataset simple ;
2. images de seulement 28x28 ;
3. images en niveaux de gris ;
4. certaines classes restent difficiles pour le CVAE, notamment `Coat` ;
5. seulement trois seeds finales ;
6. le classifieur indépendant n'est lui-même pas parfait ;
7. les métriques de diversité n'ont pas de seuil universel ;
8. le test officiel avait été consulté lors d'une phase exploratoire ;
9. les résultats ne doivent pas être extrapolés directement à des images naturelles complexes.

---

# 53. Améliorations possibles

Pistes futures :

- réseau convolutionnel plus profond ;
- beta scheduling ;
- KL annealing ;
- recherche sur latent dimension ;
- plus de seeds ;
- optimisation automatisée des hyperparamètres ;
- meilleure modélisation des classes visuellement proches ;
- FID adapté à Fashion-MNIST ;
- GAN ;
- diffusion models ;
- interpolation latente dans l'application ;
- visualisation PCA / t-SNE / UMAP du latent.

---

# 54. Questions probables du jury

## Pourquoi un VAE plutôt qu'un autoencodeur classique ?

Parce que le VAE impose une structure probabiliste au latent et rapproche la distribution apprise de `N(0,I)`. Cela rend possible la génération de nouvelles images par échantillonnage.

## À quoi servent `mu` et `logvar` ?

Ils paramètrent la distribution latente approximative `q(z|x)`.

## Qu'est-ce que le reparameterization trick ?

On écrit :

```text
z = mu + sigma * epsilon
```

pour permettre la rétropropagation malgré l'échantillonnage.

## À quoi sert le KL ?

À rapprocher la distribution latente apprise du prior `N(0,I)` et donc à régulariser l'espace latent.

## Pourquoi beta=1 ?

Parce qu'il donne le meilleur compromis entre reconstruction, régularisation, cohérence et diversité.

## Pourquoi ne pas choisir beta=4 pour le CVAE ?

Parce que le gain de cohérence n'est que de `0.72 point`, alors que beta=4 réduit la diversité et dégrade la reconstruction.

## Pourquoi trois seeds ?

Pour vérifier que les conclusions ne dépendent pas d'une seule initialisation.

## Pourquoi séparer `training_seed` et `split_seed` ?

Pour garder les mêmes données train/validation tout en changeant seulement l'aléa de l'entraînement.

## Pourquoi `decode(mu)` pour MSE et SSIM ?

Parce qu'un `z` échantillonné ajoute du bruit aux métriques de reconstruction. `decode(mu)` donne une reconstruction déterministe et comparable.

## Pourquoi garder aussi des métriques sampled ?

Parce que le VAE/CVAE est probabiliste. La loss réelle inclut un échantillonnage latent.

## Pourquoi un classifieur indépendant ?

Pour juger objectivement si les images générées correspondent à la classe demandée.

## Pourquoi mesurer la diversité uniquement sur les générations cohérentes ?

Parce qu'une image hors classe peut artificiellement augmenter la diversité. La version `COHERENT` mesure la diversité parmi les images qui respectent effectivement la condition.

## Pourquoi le CVAE a-t-il un total plus faible alors que le VAE reconstruit mieux ?

Parce que le total contient :

```text
BCE + KL
```

Le CVAE possède un KL plus faible, tandis que le VAE obtient une meilleure reconstruction déterministe.

## Le CVAE est-il meilleur que le VAE ?

Pas de manière absolue :

```text
reconstruction → avantage VAE
contrôle de classe → avantage CVAE
```

---

# 55. Ce qu'il ne faut pas dire à la soutenance

Éviter :

```text
"beta=0.1 est meilleur parce que sa loss totale est plus petite"
```

Faux, car les objectifs diffèrent selon beta.

Éviter :

```text
"82 % signifie que le CVAE est parfait"
```

Faux.

Éviter :

```text
"le CVAE est globalement meilleur que le VAE"
```

Trop simpliste.

Éviter :

```text
"0.164 est une bonne diversité parce qu'elle dépasse un seuil"
```

Il n'existe pas de seuil universel ici.

Éviter :

```text
"le test officiel n'avait jamais été vu"
```

Il avait été utilisé dans une phase exploratoire.

---

# 56. Formulations recommandées

Dire :

> Le choix final de beta=1 repose sur un compromis entre fidélité, régularisation, cohérence conditionnelle et diversité.

Dire :

> Les résultats multi-seed montrent une stabilité descriptive sur trois initialisations.

Dire :

> Le VAE possède un léger avantage en reconstruction déterministe, tandis que le CVAE apporte le contrôle conditionnel.

Dire :

> La diversité cohérente mesure la variation uniquement parmi les images qui respectent réellement la classe demandée.

Dire :

> L'évaluation finale sur le test a été réalisée après gel des choix finaux.

---

# 57. Pitch court de soutenance

> Notre projet porte sur la génération d'images Fashion-MNIST avec un VAE et un CVAE. Nous avons d'abord mis en place un protocole reproductible avec un split train/validation fixe et des seeds d'entraînement séparées. Nous avons testé plusieurs valeurs de beta et retenu beta=1 comme meilleur compromis entre reconstruction, régularisation, cohérence et diversité. Les modèles finaux ont ensuite été entraînés sur trois seeds et suivis avec MLflow. Pour le CVAE, nous avons utilisé un classifieur indépendant afin de mesurer la cohérence entre la classe demandée et l'image générée. Le CVAE final atteint 82.42 % ± 0.87 point de cohérence sur 30 000 générations. Nous avons également mesuré la diversité pixel et sémantique des générations cohérentes. Sur le test officiel, le VAE reconstruit légèrement mieux, tandis que le CVAE permet la génération contrôlée. Enfin, les checkpoints finaux ont été préparés pour l'application web.

---

# 58. Plan conseillé pour une présentation de 5 à 7 minutes

## 1. Problème

```text
Comment générer des images Fashion-MNIST
et contrôler la classe générée ?
```

## 2. Modèles

```text
VAE  → génération probabiliste
CVAE → génération probabiliste + condition de classe
```

## 3. Protocole

```text
54k train
6k validation
10k test
split_seed = 42
training seeds = 0,42,123
```

## 4. Choix de beta

```text
0.1 / 1 / 4
```

Choix final :

```text
beta = 1
```

## 5. Multi-seed

Trois entraînements finaux par architecture.

## 6. Reconstruction

Le VAE reconstruit légèrement mieux.

## 7. Génération conditionnelle

```text
CVAE coherence = 82.42 % ± 0.87 pp
```

## 8. Diversité

```text
pixel coherent    = 0.256778 ± 0.000417
semantic coherent = 0.164418 ± 0.003483
```

## 9. Conclusion

```text
VAE = reconstruction
CVAE = contrôle conditionnel
```

---

# 59. Chiffres à mémoriser

```text
Dataset
60 000 train officiel
10 000 test officiel
54 000 train final
6 000 validation

beta final
1

seeds
0, 42, 123

latent_dim
16

hidden_dim
256

batch_size
128

learning_rate
0.001

max_epochs
100

early stopping patience
10
```

Résultats clés :

```text
Classifier validation
92.62 %

CVAE coherence
82.42 % ± 0.87 pp

CVAE pixel diversity coherent
0.256778 ± 0.000417

CVAE semantic diversity coherent
0.164418 ± 0.003483
```

Test VAE :

```text
MSE  = 0.011968 ± 0.000196
SSIM = 0.687828 ± 0.001061
BCE  = 218.124 ± 0.448
```

Test CVAE :

```text
MSE  = 0.012446 ± 0.000449
SSIM = 0.684522 ± 0.005874
BCE  = 219.225 ± 0.976
```

---

# 60. Les cinq messages principaux à faire passer

1. **Le protocole a été rendu reproductible.**  
   Les seeds du split et de l'entraînement sont séparées.

2. **Le choix de beta n'a pas été fait uniquement sur la loss.**  
   Reconstruction, KL, cohérence et diversité ont été considérés.

3. **Les résultats finaux sont multi-seed.**  
   On ne présente pas un seul run chanceux.

4. **La reconstruction finale est évaluée de manière déterministe.**  
   `z = mu` pour BCE/MSE/SSIM de reconstruction.

5. **Le CVAE est évalué sur sa vraie fonction.**  
   On mesure sa cohérence conditionnelle et sa diversité, pas seulement sa reconstruction.

---

# 61. Checklist de révision avant soutenance

Être capable d'expliquer sans notes :

- Fashion-MNIST ;
- autoencodeur ;
- VAE ;
- CVAE ;
- `mu` et `logvar` ;
- reparameterization trick ;
- KL ;
- rôle de beta ;
- pourquoi beta=1 ;
- pourquoi trois seeds ;
- différence `training_seed` / `split_seed` ;
- early stopping ;
- MLflow ;
- pourquoi `decode(mu)` pour MSE/SSIM ;
- classifieur indépendant ;
- conditional accuracy ;
- diversité pixel ;
- diversité sémantique ;
- ALL vs COHERENT ;
- résultats validation ;
- résultats test ;
- point faible `Coat` ;
- pourquoi VAE reconstruit mieux ;
- pourquoi CVAE est utile pour l'application ;
- localisation des checkpoints ;
- raison de ne pas stocker les `.pt` directement dans Git.

---

# 62. Fiche ultra-courte juste avant de présenter

```text
SUJET
VAE + CVAE sur Fashion-MNIST

DATA
54k train
6k validation
10k test

PROTOCOLE
split_seed=42
training seeds=0,42,123

ARCHITECTURE
latent=16
hidden=256

BETA
0.1 / 1 / 4 testés
beta=1 retenu

POURQUOI beta=1 ?
meilleur compromis :
reconstruction + KL + cohérence + diversité

VAE
meilleure reconstruction

CVAE
génération contrôlée
cohérence = 82.42 ± 0.87 pp

DIVERSITE CVAE COHERENTE
pixel = 0.256778 ± 0.000417
semantic = 0.164418 ± 0.003483

TEST VAE
MSE  = 0.011968
SSIM = 0.687828

TEST CVAE
MSE  = 0.012446
SSIM = 0.684522

MESSAGE FINAL
VAE = reconstruction
CVAE = contrôle conditionnel
```

---

# 63. Conclusion finale

Ce projet a permis de construire un pipeline génératif complet et reproductible sur Fashion-MNIST.

Le travail couvre :

```text
compréhension théorique
→ préparation des données
→ VAE
→ CVAE
→ choix beta
→ entraînement
→ early stopping
→ MLflow
→ multi-seed
→ validation
→ reconstruction déterministe
→ cohérence conditionnelle
→ diversité
→ test officiel
→ tableaux et figures
→ Git
→ livraison des modèles à l'application
```

Le résultat final principal est :

```text
VAE
→ meilleure reconstruction déterministe

CVAE
→ génération conditionnelle contrôlée
→ cohérence 82.42 % ± 0.87 pp
→ diversité stable
```

Le choix de `beta=1` est justifié comme meilleur compromis entre reconstruction, régularisation et qualité générative.

Les checkpoints finaux sont prêts à être utilisés par l'application web, et les résultats scientifiques sont consolidés dans les fichiers de rapport.

**Le projet expérimental Fashion-MNIST VAE/CVAE est terminé.**

---

# 64. Mise à jour — Packaging et déploiement MLflow

> Cette section documente le travail réalisé **après** la clôture scientifique du projet (section 63), en réponse à une nouvelle consigne : packager et déployer les modèles avec MLflow, et fournir des endpoints utilisables par l'application web. Rien dans les sections précédentes n'a été modifié ; ce qui suit s'ajoute au travail déjà terminé.

## Consigne reçue

> Packager nos modèles avec MLflow, déployer nos modèles avec MLflow, et fournir les endpoints pour l'application web.

Jusqu'ici, MLflow n'avait été utilisé que pour le **tracking** (section 11) : suivi des hyperparamètres, métriques, et artefacts (checkpoints, historiques CSV) au cours des entraînements. Il restait à transformer les checkpoints finaux en modèles réellement **exploitables via une interface HTTP**, sans toucher à l'entraînement ni aux checkpoints déjà figés.

## Principe : ne pas retoucher aux modèles figés

Conformément à la section 40, les checkpoints utilisés pour ce travail sont exactement ceux déjà retenus pour l'application :

```text
vae_beta_1_seed42_final.pt
cvae_beta_1_seed42_final.pt
```

Aucun réentraînement, aucun changement de beta ou de seed n'a eu lieu à cette étape.

---

# 65. Nouveau dossier `deployment/`

Un nouveau dossier a été créé à la racine du sous-projet, séparé de `evaluation/` (qui mesure les modèles) :

```text
deployment/
├── __init__.py
├── mlflow_models.py           # wrappers mlflow.pyfunc pour VAE et CVAE
├── register_models.py         # empaquetage + enregistrement Model Registry
├── export_standalone_models.py# export en dossiers portables
├── test_serving.py            # tests automatiques des endpoints
├── packaged_models/           # modèles exportés, prêts à transmettre
│   ├── vae_model/
│   └── cvae_model/
└── README.md
```

## `mlflow_models.py`

Contient deux classes `mlflow.pyfunc.PythonModel` (`VAEPyfuncModel`, `CVAEPyfuncModel`) qui **réutilisent directement** les classes existantes `models/vae.py` et `models/cvae.py`, sans réécrire l'architecture. Chaque wrapper :

- charge le checkpoint `.pt` correspondant au moment de `load_context()` ;
- reconstruit les tenseurs latents `z` à partir d'une requête JSON (`z` fourni ou tiré aléatoirement) ;
- appelle `model.decode(z)` (VAE) ou `model.decode(z, labels)` (CVAE) ;
- encode l'image générée en PNG base64, directement affichable côté application web.

## `register_models.py`

Charge les deux checkpoints figés, les empaquette avec `mlflow.pyfunc.log_model(...)`, puis les enregistre dans le MLflow Model Registry, en réutilisant le même backend que l'entraînement (`mlflow.db`, `mlartifacts/`, section 11) :

```text
FashionMNIST-VAE   version 1
FashionMNIST-CVAE  version 1
```

## Service local et tests

Les modèles ont été servis avec :

```text
mlflow models serve -m "models:/FashionMNIST-VAE/1"  -p 5001 --env-manager local
mlflow models serve -m "models:/FashionMNIST-CVAE/1" -p 5002 --env-manager local
```

Puis validés avec `test_serving.py`, qui vérifie automatiquement :

- que le VAE génère une image 28x28 valide ;
- que le CVAE génère correctement une image pour **chacune des 10 classes** Fashion-MNIST ;
- que le CVAE est déterministe (même `z` + même classe → image strictement identique).

Résultat : tous les tests sont passés avec succès.

---

# 66. Export en dossiers portables et transmission à l'application

Le travail se faisant en parallèle via Git (chaque membre sur sa machine), les serveurs `mlflow models serve` locaux et le Model Registry local ne sont pas transmissibles tels quels à un coéquipier.

`export_standalone_models.py` a été ajouté pour résoudre ce point : il exporte chaque modèle enregistré vers un dossier **autonome et portable** (`deployment/packaged_models/vae_model/`, `.../cvae_model/`), contenant tout ce qu'il faut pour être rechargé ailleurs, sans registre ni connexion à la machine d'origine :

```text
vae_model/
├── MLmodel
├── python_model.pkl
├── artifacts/vae_beta_1_seed42_final.pt
├── code/models/, code/deployment/
├── requirements.txt
└── input_example.json
```

Ces deux dossiers ont été compressés (`packaged_models.zip`, ~42 Mo) et transmis directement au collègue responsable de l'application web.

---

# 67. Architecture finale retenue côté application web

Après discussion d'équipe, l'application web centralise l'accès aux modèles des **trois datasets** (MNIST, Fashion-MNIST, CelebA) derrière **un seul port FastAPI**, plutôt que d'exposer un port MLflow différent par modèle.

## Ce qui a été écarté

Faire tourner `mlflow models serve` en permanence pour chacun des 6 modèles (2 par dataset), ce qui aurait nécessité 6 processus actifs en parallèle et 6 ports différents à maintenir manuellement.

## Ce qui a été retenu

FastAPI charge directement les modèles packagés en mémoire, une seule fois au démarrage, avec `mlflow.pyfunc.load_model(chemin_local)` — en réutilisant exactement les dossiers `packaged_models/` déjà exportés :

```text
Application Web
      │
      ▼
FastAPI (port unique, ex. 8000)
   /generate  { "dataset": ..., "model_type": ..., "target_class": ..., "z": ... }
      │
      ├── MNIST-VAE / MNIST-CVAE
      ├── FashionMNIST-VAE / FashionMNIST-CVAE   ← ce sous-projet
      └── CelebA-VAE / CelebA-CVAE
```

Le champ `"dataset"` dans le corps de la requête sélectionne, côté FastAPI, quel modèle utiliser — le routage se fait en mémoire, pas via des ports séparés. Cette architecture reste compatible avec tout ce qui a été packagé en section 65 : aucun changement n'a été nécessaire côté `deployment/`, seul le mode de chargement change chez le collègue (`load_model` au lieu de `models serve`).

Un squelette `main.py` FastAPI correspondant a été fourni au collègue en charge du backend/frontend, avec un endpoint `/health` permettant de vérifier rapidement combien de modèles (sur 6 attendus) sont effectivement chargés.

---

# 68. Publication sur GitHub — mise à jour de la politique de livraison

Les sections 41 et 47 (non modifiées ci-dessus) reflétaient l'état du projet à ce moment-là : les poids `.pt` étaient volontairement exclus de Git (`.gitignore`) et transmis uniquement par zip/Drive, GitHub n'étant pas jugé nécessaire pour la distribution des poids.

**Cette politique évolue à partir de maintenant** : le professeur souhaite pouvoir vérifier directement sur GitHub le travail de packaging et de déploiement MLflow, y compris les modèles eux-mêmes. En conséquence, en plus du code, sont désormais versionnés dans le dépôt :

```text
checkpoints/
├── vae_beta_1_seed42_final.pt
└── cvae_beta_1_seed42_final.pt

deployment/
├── mlflow_models.py
├── register_models.py
├── export_standalone_models.py
├── test_serving.py
├── README.md
└── packaged_models/
    ├── vae_model/
    └── cvae_model/
```

Le `.gitignore` (section 47) a été ajusté pour ne plus exclure ces fichiers précis, tout en continuant d'exclure l'environnement virtuel, `mlflow.db` et `mlartifacts/` (le backend MLflow local complet, volumineux et non nécessaire à la vérification du travail — le Model Registry et l'historique complet des runs ne sont pas requis, seuls le code et les modèles packagés le sont).

Ce choix reste cohérent avec les bonnes pratiques : on ne verse pas l'intégralité de la base de tracking (qui grossit à chaque run, y compris les runs de test), mais on verse le résultat final consommable (modèles packagés + code de déploiement), exactement ce dont le professeur a besoin pour vérifier le travail.

---

# 69. Conclusion mise à jour

Le projet couvre désormais, en plus de la chaîne expérimentale de la section 63 :

```text
... → livraison des modèles à l'application (section 63)
      → packaging MLflow (mlflow.pyfunc)
      → enregistrement Model Registry
      → déploiement (mlflow models serve)
      → tests automatiques des endpoints (10 classes + déterminisme)
      → export en dossiers portables
      → transmission au collègue applicatif
      → publication du code et des modèles sur GitHub
```

**Le volet packaging et déploiement MLflow du projet Fashion-MNIST VAE/CVAE est terminé et vérifiable sur GitHub.**
