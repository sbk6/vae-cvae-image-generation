# Document de présentation au groupe — VAE / CVAE sur MNIST

> Ce document est écrit pour être **lu à voix haute** à l'équipe : il explique ce qui a été fait, comment, avec quelles valeurs et pourquoi. Les chiffres exacts et les fichiers à montrer à l'écran sont dans le [README.md](../README.md) ; ce document sert de script.

---

## 1. Pour resituer le projet

On travaille sur le sujet **"VAE conditionnel pour la génération d'images"**. L'idée : construire un modèle qui apprend à partir d'images, puis qui peut en générer de nouvelles. On fait ça en deux versions :

- un **VAE** classique, qui génère des images "libres" (on ne choisit pas quoi il produit),
- un **CVAE**, qui est la version où on peut **demander une classe précise** (par exemple : "génère-moi un 7").

L'énoncé demande de faire ça sur trois jeux de données (MNIST, Fashion-MNIST, CelebA). Pour l'instant, on a travaillé uniquement sur **MNIST** (les chiffres manuscrits de 0 à 9), pour valider toute la chaîne avant de passer aux datasets plus compliqués.

## 2. Ce qu'on a fait, dans l'ordre

### Étape 1 — Le socle du projet
Avant cette partie du travail : mise en place du projet (structure des dossiers, lecture de la config depuis des fichiers YAML, chargement de MNIST), implémentation du VAE, du CVAE, et de la fonction de perte. Ça avait été testé mais seulement en mode "test rapide" (quelques minutes, pour vérifier que le code tourne sans planter), donc les résultats de cette étape n'étaient pas représentatifs.

### Étape 2 — On a trouvé un bug important
En reprenant le projet, on s'est rendu compte que le VAE et le CVAE **sauvegardaient leur modèle entraîné dans le même fichier**. Résultat concret : le dernier entraînement lancé (qui était un test rapide du CVAE, donc peu entraîné) avait **écrasé** le VAE qui, lui, avait été correctement entraîné avant. Du coup, les images qu'on montrait comme résultat du CVAE venaient en fait d'un modèle qui n'avait quasiment rien appris.

On a corrigé ça en donnant à **chaque entraînement son propre dossier de sortie**. Maintenant, impossible qu'un modèle en écrase un autre par accident.

### Étape 3 — On a vraiment entraîné les deux modèles

On tourne uniquement sur CPU (pas de carte graphique disponible), donc un passage complet sur les 54 000 images d'entraînement de MNIST (une "epoch") prend environ **2 minutes 30 à 3 minutes**. On a choisi d'entraîner chaque modèle sur **10 epochs**, ce qui représente 25 à 30 minutes par modèle. On a lancé le VAE et le CVAE **en parallèle** (deux processus en même temps) pour gagner du temps.

**Pourquoi 10 epochs et pas plus ?** En regardant l'évolution de la perte à chaque epoch, on voit qu'elle baisse fort au début puis se stabilise clairement à partir de l'epoch 7-8 (le gain entre l'epoch 9 et l'epoch 10 est minime). Continuer plus longtemps aurait coûté beaucoup de temps de calcul pour un gain faible — un choix de compromis assumé, pas une contrainte technique dure.

**Les réglages utilisés pour ces deux modèles principaux :**
- `latent_dim = 16` : la taille de l'espace latent (la "représentation compressée" de l'image). 16 est une valeur standard pour un dataset aussi simple que MNIST — assez grand pour capturer la variabilité des 10 classes de chiffres, assez petit pour forcer le modèle à vraiment compresser l'information plutôt que de "tricher".
- `hidden_channels = 32` : la largeur des couches convolutives de l'encodeur/décodeur — un choix standard pour rester rapide à entraîner sur CPU tout en ayant assez de capacité pour MNIST (images 28×28 en niveaux de gris, un problème relativement simple).
- `batch_size = 128` : le nombre d'images traitées ensemble à chaque étape — valeur courante qui équilibre vitesse et stabilité de l'entraînement.
- `lr = 0.001` (taux d'apprentissage, avec l'optimiseur Adam) : la valeur par défaut la plus utilisée pour Adam, un bon point de départ qui converge de façon fiable sans qu'on ait eu besoin de la régler finement pour ce projet.
- `beta = 1.0` : le poids donné à la régularisation de l'espace latent (voir étape 4 ci-dessous pour l'explication complète). On a choisi cette valeur *après* avoir fait l'étude d'ablation qui suit, précisément parce que l'ablation a montré que c'était le meilleur compromis.

### Étape 4 — L'étude d'ablation sur β (demandée explicitement par l'énoncé)

Il faut d'abord comprendre à quoi sert β. La perte qu'on minimise a deux parties : `loss = reconstruction + β × KL`.
- La partie **reconstruction** pousse le modèle à bien reproduire l'image de départ.
- La partie **KL** pousse l'espace latent à rester "régulier" (proche d'une distribution normale), ce qui est ce qui permet ensuite de générer de nouvelles images en tirant un point au hasard dans cet espace.

β est le curseur entre ces deux objectifs. **On a testé 3 valeurs : 0.1, 1.0 et 5.0** — volontairement très écartées les unes des autres (un facteur 50 entre la plus petite et la plus grande) pour bien voir l'effet aux deux extrêmes, plutôt que de tester des valeurs proches qui auraient donné des résultats difficiles à distinguer.

Pour aller plus vite (3 entraînements au lieu d'un seul, sur CPU), on a réduit ces runs d'ablation à **6 epochs** et à un **sous-ensemble de 12 000 images** d'entraînement (au lieu des 54 000). Important : la validation, elle, reste toujours faite sur l'ensemble complet, donc la comparaison entre les 3 valeurs de β reste fiable.

**Ce qu'on a observé, et c'est le résultat le plus parlant de la séance :**
- Avec **β = 0.1** : très bonne reconstruction, mais l'espace latent devient peu régulier (mesuré par un KL qui monte à 39, très haut).
- Avec **β = 5.0** : le KL s'effondre quasiment à zéro. Ça veut dire que le modèle a **arrêté d'utiliser l'espace latent** — c'est un phénomène connu qui s'appelle le "posterior collapse". On le voit très clairement sur une image de comparaison qu'on a générée : à β=5.0, les reconstructions deviennent des taches grises informes, presque identiques peu importe l'image de départ.
- Avec **β = 1.0** : le compromis le plus équilibré entre les deux — c'est pour ça qu'on a retenu cette valeur pour nos modèles principaux.

### Étape 5 — Visualiser l'espace latent

On a projeté l'espace latent en 2D avec une technique appelée **t-SNE**, pour le VAE et pour le CVAE, en coloriant chaque point selon le vrai chiffre qu'il représente.

**Résultat surprenant et intéressant à expliquer au groupe :** pour le VAE, les points se regroupent nettement par couleur — **alors que le VAE n'a jamais reçu l'information de classe pendant l'entraînement**. Le modèle a appris tout seul à séparer les chiffres dans son espace latent, simplement parce que c'est la meilleure stratégie pour bien reconstruire des images très différentes. Pour le CVAE, à l'inverse, les couleurs sont mélangées : logique, puisque le CVAE reçoit déjà la classe séparément, il n'a plus besoin de la coder dans l'espace latent, qui se concentre alors sur autre chose (le style d'écriture : inclinaison, épaisseur du trait...).

### Étape 6 — Interpolation

On a aussi vérifié qu'on peut "voyager" dans l'espace latent : en partant du point qui correspond à un vrai `3`, et en allant progressivement vers le point qui correspond à un vrai `8`, on décode chaque étape intermédiaire. Le résultat montre une transition douce, sans saut brutal — la preuve que l'espace latent appris est continu et bien structuré, pas juste un ensemble de points isolés.

### Étape 7 — Comparaison chiffrée VAE vs CVAE

Sur les 10 000 images du test set : le VAE et le CVAE reconstruisent quasiment aussi bien l'un que l'autre. La différence n'est donc pas là. La vraie différence, c'est que **seul le CVAE permet de choisir la classe qu'on veut générer**. On a essayé de mesurer ça automatiquement avec une méthode simple (comparer chaque image générée à "l'image moyenne" de chaque classe), et on est tombé sur un résultat bas qui contredisait ce qu'on voyait à l'œil sur les grilles d'images. On a creusé pourquoi (voir section suivante) plutôt que de laisser ce chiffre trompeur tel quel.

## 3. Difficultés rencontrées et comment on les a résolues

**Le bug du checkpoint partagé** (déjà expliqué à l'étape 2) : résolu en séparant les dossiers de sortie par expérience.

**La lenteur du CPU** : un entraînement complet prend 25-45 minutes. On a géré ça en limitant le nombre d'epochs pour les modèles principaux, en réduisant le volume de données pour l'ablation (où l'on répète l'entraînement 3 fois), et en faisant tourner plusieurs entraînements en parallèle.

**Un chiffre de mesure automatique qui ne collait pas avec ce qu'on voyait** : on a testé si le CVAE générait bien la bonne classe en utilisant une méthode de comparaison "au pixel près" avec l'image moyenne de chaque classe. Cette méthode a donné un score bas (autour de 29%), alors que les grilles d'images montraient clairement que la plupart des classes étaient bien générées. Plutôt que d'ignorer cette contradiction, on a vérifié la méthode sur de vraies images (elle donne 82% sur des vrais chiffres, donc elle fonctionne en soi) et compris que le problème vient du **flou** des images générées par un VAE : cette méthode de comparaison est très sensible au flou, en particulier pour les chiffres fins comme le 1, le 4, le 7 ou le 9. Conclusion : c'est notre méthode de mesure qui est limitée, pas forcément le CVAE qui contrôle mal — l'inspection visuelle reste, pour l'instant, la preuve la plus fiable.

## 4. Où sont les résultats concrets à montrer

- `reports/figures/cvae_grid.png` : une ligne par chiffre demandé (0 à 9) — l'image la plus parlante pour montrer que "ça marche".
- `reports/figures/latent_tsne_vae.png` et `latent_tsne_cvae.png` : la différence d'organisation de l'espace latent entre les deux modèles.
- `reports/figures/interpolation_vae_3_to_8.png` : la transition progressive entre deux chiffres.
- `reports/figures/ablation_beta_reconstruction_comparison.png` : l'effet visuel de β sur la qualité de reconstruction, du très net (β=0.1) au collapse complet (β=5.0).
- `docs/RESULTATS.md` : le tableau chiffré de l'ablation.
- Le [README.md](../README.md) : le document complet, avec chaque résultat expliqué et sourcé.

## 5. Ce qu'on fait ensuite

- Le professeur a demandé qu'on suive les entraînements avec **MLflow** (un outil qui enregistre automatiquement les paramètres et les métriques de chaque entraînement, consultable dans une interface web) — c'est en cours d'intégration.
- Brancher réellement **Fashion-MNIST** (le code a un point d'entrée prévu mais qui charge encore MNIST pour l'instant, ce n'est pas encore fait).
- Intégrer **CelebA**.
- Remplacer notre mesure "au pixel près" par un vrai petit classifieur, pour avoir un chiffre de contrôlabilité plus fiable.
- Préparer la démonstration web (choisir une classe → génération d'image).

## 6. Résumé en une minute, si on doit faire très court

"On a repris le projet, trouvé et corrigé un bug qui faisait qu'un de nos modèles n'était pas vraiment entraîné, puis on a vraiment entraîné un VAE et un CVAE sur MNIST. On a testé 3 valeurs de β (0.1, 1.0, 5.0) pour voir leur effet sur la qualité — 1.0 est le meilleur compromis, 5.0 fait s'effondrer l'espace latent. On a visualisé l'espace latent : il se structure tout seul par classe pour le VAE, mais pas pour le CVAE, ce qui explique pourquoi le CVAE permet de choisir la classe générée. On a essayé de mesurer automatiquement si le CVAE contrôle bien la génération, trouvé un chiffre trompeur, et compris pourquoi plutôt que de le cacher. La suite : Fashion-MNIST et CelebA."

> Mise à jour depuis l'écriture de ce script : MLflow est maintenant pleinement intégré (tous les entraînements sont trackés), et l'étude d'ablation ainsi que les modèles principaux ont été validés sur 3 seeds différents pour confirmer que les résultats sont reproductibles. Détails à jour dans le [README.md](../README.md).

## 7. Questions possibles et réponses courtes

**Pourquoi commencer par MNIST ?** Dataset simple et rapide, permet de valider toute la chaîne (données → modèle → loss → entraînement → visualisation) avant de passer à des images plus complexes.

**Quelle est la vraie différence entre VAE et CVAE, au-delà du code ?** Le VAE encode l'identité de la classe *dans* l'espace latent, de façon non supervisée (on le voit sur le t-SNE). Le CVAE reçoit la classe séparément, donc son espace latent se spécialise sur autre chose (le style). Le CVAE permet donc de *choisir* la classe générée, le VAE non.

**Quel est le meilleur β trouvé ?** β = 1.0, meilleur compromis entre fidélité de reconstruction et régularité de l'espace latent, confirmé sur 3 seeds différents. β=0.1 reconstruit mieux mais régularise mal ; β=5.0 régularise trop et fait s'effondrer l'espace latent (posterior collapse), visible directement sur les images.

**Le CVAE contrôle-t-il vraiment bien la génération ?** Visuellement oui (`cvae_grid.png`), mais notre première tentative de mesure automatique donnait un chiffre bas et trompeur — nous avons identifié pourquoi (sensibilité au flou de la méthode utilisée) plutôt que de le cacher.

**Quel est le principal blocage actuel ?** Aucun blocage bloquant : la suite dépend surtout de temps de calcul (Fashion-MNIST et CelebA demandent les mêmes étapes que MNIST).
