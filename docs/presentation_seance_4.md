# Présentation de séance 4 — VAE / CVAE sur MNIST : vrais entraînements, ablation, comparaison

Bonsoir Monsieur,

Nous continuons sur le sujet **VAE conditionnel pour la génération d'images**. Lors de la séance précédente, nous avions posé le socle du projet (structure, chargement MNIST, VAE, CVAE, loss ELBO) mais uniquement testé en mode rapide ("smoke-test"). Cette séance, nous avons **réellement entraîné** les deux modèles, réalisé les livrables demandés par l'énoncé (ablation sur β, visualisation de l'espace latent, interpolation, comparaison quantitative), et corrigé un bug qui faussait nos résultats précédents.

*(Pour tous les détails, chiffres exacts et fichiers, voir le [README.md](../README.md) à la racine du projet — ce document-ci est la version condensée pour l'oral.)*

## 1. Ce que nous avons fait depuis la dernière séance

- Trouvé et corrigé un bug : le VAE et le CVAE partageaient le même fichier de checkpoint, donc le dernier entraînement lancé (un test rapide du CVAE) avait écrasé le VAE correctement entraîné. Chaque expérience a maintenant son propre dossier.
- Entraîné le **VAE** pendant 10 epochs sur les 60 000 images de MNIST.
- Entraîné le **CVAE** dans les mêmes conditions.
- Réalisé l'**étude d'ablation** demandée par l'énoncé sur le poids β (3 valeurs : 0.1, 1.0, 5.0).
- Produit la **visualisation de l'espace latent** en 2D (t-SNE) pour les deux modèles.
- Produit une **interpolation** entre deux exemples de classes différentes dans l'espace latent.
- Fait une **comparaison quantitative** VAE vs CVAE (reconstruction, KL, contrôlabilité).

## 2. Où se trouve chaque chose (pour pouvoir montrer en direct)

- `reports/experiments/vae_main/`, `reports/experiments/cvae_main/` : logs (`training_log.csv`) et modèle entraîné (`best_checkpoint.pth`) des deux modèles principaux.
- `reports/experiments/ablation/beta_0.1/`, `beta_1.0/`, `beta_5.0/` : un dossier par valeur de β testée.
- `reports/figures/` : toutes les images (voir tableau détaillé dans le README, section 6).
- `docs/RESULTATS.md` : tableau d'ablation généré automatiquement par `scripts/run_ablation.py`.
- `scripts/run_ablation.py`, `scripts/evaluate.py` : les deux scripts qui produisent les livrables demandés par l'énoncé.
- `src/visualization/latent.py`, `src/visualization/interpolation.py` : le code des visualisations.

## 3. Répartition du travail dans le groupe de 4

Chaque membre garde un rôle principal, mais tout le monde relit et challenge le travail des autres avant validation :

- **Membre 1** — données et socle technique (chargement MNIST, configuration YAML).
- **Membre 2** — modèles (VAE, CVAE, loss ELBO).
- **Membre 3** — entraînement et expériences (lancement des runs, étude d'ablation, évaluation chiffrée).
- **Membre 4** — visualisation et restitution (figures, documentation, présentation orale).

## 4. Résultats obtenus

### 4.1 Les modèles génèrent-ils correctement ?

`reports/figures/cvae_grid.png` : une ligne par classe demandée (0 à 9). La plupart des lignes sont clairement reconnaissables — le CVAE contrôle bien la classe générée dans la majorité des cas.

`reports/figures/vae_random_samples_grid.png` : génération libre du VAE (pas de condition). On observe que certaines classes dominent largement, d'autres n'apparaissent presque jamais — normal, rien ne force le VAE classique à couvrir toutes les classes uniformément.

### 4.2 L'espace latent est-il bien structuré ?

`reports/figures/latent_tsne_vae.png` vs `reports/figures/latent_tsne_cvae.png` : le résultat le plus intéressant de la séance. Dans le VAE, les points se regroupent nettement par classe **alors que le label n'a jamais été donné au modèle**. Dans le CVAE, les classes sont mélangées : logique, puisque le CVAE reçoit déjà la classe séparément, son espace latent code plutôt le style d'écriture.

`reports/figures/interpolation_vae_3_to_8.png` et `interpolation_vae_1_to_7.png` : transition progressive entre deux chiffres, sans saut brutal — l'espace latent est continu.

### 4.3 L'étude d'ablation sur β

| β | Reconstruction (val) | KL (val) |
|---|---|---|
| 0.1 | 680.87 | 39.47 |
| 1.0 | 691.83 | 15.37 |
| 5.0 | 725.11 | 0.56 |

À β=5.0, le KL s'effondre presque à zéro : le modèle cesse d'utiliser l'espace latent ("posterior collapse"), visible très nettement sur `reports/figures/ablation_beta_reconstruction_comparison.png` (les reconstructions deviennent des taches grises informes). **β=1.0 est le meilleur compromis** que nous avons trouvé entre qualité de reconstruction et régularité de l'espace latent.

### 4.4 VAE vs CVAE, chiffré

Sur le test set complet (10 000 images) : reconstruction quasi identique entre les deux modèles (677 pour le VAE, 677 pour le CVAE). La vraie différence n'est pas la qualité de reconstruction, mais la **contrôlabilité** : seul le CVAE permet de choisir la classe générée.

## 5. Difficultés rencontrées et comment nous les avons résolues

**Bug de checkpoint partagé** : VAE et CVAE écrivaient dans le même fichier. Résolu en donnant un dossier de sortie dédié à chaque expérience.

**Entraînement lent (CPU uniquement)** : ~2min30-3min par epoch sur les 54 000 images. Résolu en limitant les modèles principaux à 10 epochs (suffisant pour une convergence nette) et en réduisant le train set à 12 000 images pour l'étude d'ablation uniquement (la validation, elle, reste complète).

**Une mesure automatique de contrôlabilité du CVAE contredisait ce qu'on voyait à l'œil** (29% de précision mesurée, alors que la grille d'images semblait clairement correcte). Plutôt que d'ignorer cette contradiction, nous avons vérifié la méthode sur de vrais chiffres (82% de précision, donc la méthode fonctionne en soi) et compris que le problème vient de la sensibilité de cette méthode au flou des images générées par un VAE — en particulier pour les chiffres fins (1, 4, 7, 9). Ce chiffre sous-estime donc la vraie contrôlabilité du CVAE ; l'inspection visuelle reste, à ce stade, la preuve la plus fiable. Détails dans le README, section 6.4.

## 6. Ce que nous comptons faire ensuite

- Brancher réellement Fashion-MNIST (le point d'extension existe dans le code mais charge encore MNIST à ce stade).
- Intégrer CelebA (ou un sous-échantillon), avec conditionnement multi-attributs.
- Remplacer notre mesure de contrôlabilité approximative par un vrai petit classifieur entraîné sur MNIST.
- Préparer la démonstration web demandée par l'énoncé.
- Si possible, relancer les modèles principaux avec plus d'epochs sur une machine avec GPU.

## 7. Questions possibles du professeur et réponses courtes

**Q : Quelle est la vraie différence entre VAE et CVAE, au-delà du code ?**
R : Le VAE encode l'identité de la classe *dans* l'espace latent, de façon non supervisée. Le CVAE reçoit la classe séparément, donc son latent se spécialise sur autre chose (le style). Seul le CVAE permet de choisir la classe générée.

**Q : Quel est le meilleur β trouvé ?**
R : β=1.0. β=0.1 reconstruit mieux mais régularise mal le latent ; β=5.0 fait s'effondrer l'espace latent (posterior collapse), visible directement sur les images.

**Q : Le CVAE contrôle-t-il vraiment bien la génération ?**
R : Visuellement oui. Notre première mesure automatique donnait un chiffre trompeur — nous avons identifié pourquoi plutôt que de le cacher (voir section 5).

**Q : Quel est le principal blocage actuel ?**
R : Aucun blocage réel ; la suite dépend surtout du temps de calcul disponible (Fashion-MNIST et CelebA demandent les mêmes étapes que MNIST, mais chaque entraînement prend 30 à 45 minutes sur CPU).

## 8. Message de synthèse oral

"Bonsoir Monsieur, cette séance nous avons corrigé un bug qui faisait qu'un de nos modèles précédents n'était en réalité pas entraîné, puis nous avons réellement entraîné un VAE et un CVAE sur MNIST. Nous avons mené l'étude d'ablation demandée sur β : β=1.0 offre le meilleur compromis, β=5.0 fait s'effondrer l'espace latent, ce qu'on voit très nettement sur nos figures. Nous avons aussi visualisé l'espace latent en 2D : il se structure tout seul par classe pour le VAE, mais pas pour le CVAE, ce qui illustre bien pourquoi le CVAE permet de contrôler la génération. Nous avons essayé de mesurer automatiquement cette contrôlabilité, obtenu un chiffre qui contredisait ce qu'on voyait à l'œil, et creusé pour comprendre pourquoi plutôt que de l'ignorer. Les prochaines étapes sont Fashion-MNIST, CelebA, et une vraie mesure de contrôlabilité avec un classifieur entraîné."
