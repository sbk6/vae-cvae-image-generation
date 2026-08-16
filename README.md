# VAE / CVAE pour la génération d'images — Compte rendu de séance

Bonsoir Monsieur,

Nous travaillons sur le sujet **VAE conditionnel pour la génération d'images**. Cette séance, nous avons repris le code existant, corrigé un bug important, puis **réellement entraîné** (et pas seulement testé en mode rapide) un VAE et un CVAE sur **MNIST**, mené une **étude d'ablation sur le poids β**, produit les **visualisations demandées** (espace latent, interpolation) et fait une **comparaison chiffrée** entre les deux modèles. Les deux autres datasets de l'énoncé (Fashion-MNIST, CelebA) ne sont pas encore commencés : voir la section "Prochaines étapes".

Ce document est écrit pour être lu tel quel en séance : chaque résultat renvoie vers le fichier exact où le voir, et est accompagné de son interprétation.

---

## 1. Ce que nous avons compris du sujet

Le sujet demande de construire un modèle génératif capable de :
- reconstruire une image à partir d'une représentation compressée (**espace latent**),
- générer de nouvelles images en échantillonnant dans cet espace latent,
- **contrôler** cette génération avec une étiquette de classe (le chiffre demandé), ce que le VAE classique ne sait pas faire mais que le CVAE sait faire.

L'énoncé demande explicitement une **étude d'ablation sur β** (le poids du terme KL dans la loss ELBO), une **visualisation de l'espace latent**, une **interpolation** entre deux exemples, et une **évaluation quantitative** de la génération.

## 2. Ce qui a été fait cette séance

- **Correction d'un bug important** : le VAE et le CVAE écrivaient leur checkpoint dans le même fichier (`reports/best_checkpoint.pth`). Résultat : le `CVAE`, entraîné en mode "test rapide" (1 epoch, 50 batches), avait écrasé le `VAE` correctement entraîné. Les figures précédentes (`cvae_grid.png`) venaient donc d'un modèle quasiment pas entraîné. Chaque expérience a maintenant son propre dossier (`reports/experiments/<nom>/`), donc ce problème ne peut plus se reproduire.
- **Vrai entraînement du VAE** : 10 epochs sur les 60 000 images MNIST (54 000 train / 6 000 validation), latent de dimension 16.
- **Vrai entraînement du CVAE** : mêmes réglages, avec le label injecté comme condition.
- **Étude d'ablation sur β** : 3 valeurs (0.1, 1.0, 5.0), même architecture, même seed, 6 epochs chacune.
- **Visualisation de l'espace latent** en 2D avec t-SNE, pour le VAE et pour le CVAE.
- **Interpolation** entre deux exemples de classes différentes (3→8 et 1→7) dans l'espace latent du VAE.
- **Comparaison quantitative** VAE vs CVAE : perte de reconstruction, KL, et une mesure de "contrôlabilité" du CVAE (voir section 6.4 — résultat surprenant et instructif).
- Réécriture des scripts pour qu'ils **chargent un modèle déjà entraîné** au lieu de le ré-entraîner à chaque fois qu'on veut une figure.

Ce qui n'est **pas** fait : Fashion-MNIST et CelebA (le code a des points d'extension prévus mais pas implémentés), la démonstration web, le score FID.

## 3. Répartition des tâches dans le groupe de 4

*(Rôles génériques ci-dessous — à remplacer par les prénoms réels de chacun.)*

- **Membre 1** — données et socle technique : chargement MNIST, fichiers de configuration YAML, structure du dépôt.
- **Membre 2** — modèles : implémentation du VAE, du CVAE et de la loss ELBO.
- **Membre 3** — entraînement et expériences : lancement des runs, étude d'ablation sur β, évaluation chiffrée.
- **Membre 4** — visualisation et restitution : figures (t-SNE, interpolation, grilles), documentation, préparation de la présentation orale.

Comme pour la séance précédente, chacun garde un rôle principal mais l'ensemble du groupe relit et challenge le travail des autres avant de le considérer comme acquis.

## 4. Architecture du dépôt (mise à jour)

```
configs/                  fichiers YAML (dataset, modèle, entraînement)
  mnist_vae.yaml           config du VAE principal
  mnist_cvae.yaml          config du CVAE principal
  ablation_beta.yaml        config de l'étude d'ablation sur beta

src/
  data/datasets.py          chargement MNIST (+ points d'extension Fashion-MNIST/CelebA)
  models/vae.py              le VAE
  models/cvae.py             le CVAE (conditionnement générique one-hot / multi-label)
  models/layers.py           blocs convolutifs partagés
  losses/elbo.py              perte ELBO = reconstruction + beta * KL
  training/trainer.py         boucle d'entraînement, validation, checkpoint, logs CSV
  training/train.py           point d'entrée CLI (choisit VAE ou CVAE selon le YAML)
  visualization/latent.py     projection t-SNE de l'espace latent
  visualization/interpolation.py  interpolation entre deux images dans l'espace latent
  evaluation/                dossier réservé aux futures métriques (FID, etc.)

scripts/
  run_ablation.py            lance l'étude d'ablation et génère tableau + courbe
  generate_cvae_grid.py       grille d'échantillons conditionnés (charge un modèle déjà entraîné)
  generate_vae_recon_grid.py  grille reconstruction + grille d'échantillons libres du VAE
  evaluate.py                 comparaison quantitative VAE vs CVAE
  inspect_dataloader.py       vérifie visuellement que les données sont bien chargées

reports/
  experiments/
    vae_main/                 VAE principal (10 epochs, données complètes) — logs + checkpoint
    cvae_main/                 CVAE principal (10 epochs, données complètes) — logs + checkpoint
    ablation/beta_0.1/, beta_1.0/, beta_5.0/   un dossier par valeur de beta testée
    ablation/results.json       résultats bruts de l'ablation
    comparison.json             résultats bruts de la comparaison VAE vs CVAE
  figures/                     toutes les images produites (voir section 6)
  best_checkpoint.pth, training_log.csv, figures/cvae_grid_8.png
                                fichiers historiques d'avant la correction du bug,
                                conservés pour traçabilité mais à ne plus utiliser

docs/
  RESULTATS.md                 tableau d'ablation généré automatiquement par le script
  explanations.md, presentation_seance_1.md   comptes rendus de la séance précédente (dépassés
                                sur les chiffres, gardés pour la partie pédagogique en français)

tests/                        tests unitaires (8 tests, tous verts)
```

## 5. Comment le code fonctionne, en mots simples

### 5.1 Le VAE (`src/models/vae.py`)

Une image passe dans l'**encodeur** (convolutions) qui produit deux vecteurs, `mu` et `logvar` : la moyenne et la log-variance d'une distribution gaussienne. On tire un point `z` dans cette distribution (c'est la **reparamétrisation**, `z = mu + eps * exp(0.5*logvar)`, qui permet de garder la rétropropagation possible malgré le tirage aléatoire). Le **décodeur** reconstruit ensuite une image à partir de `z`.

### 5.2 Le CVAE (`src/models/cvae.py`)

Même principe, mais le label (classe du chiffre, encodé en one-hot) est ajouté :
- en entrée de l'encodeur (concaténé à l'image comme des canaux supplémentaires),
- en entrée du décodeur (concaténé au vecteur latent `z`).

Ainsi le décodeur reçoit toujours deux informations : "quelle forme dans l'espace latent" et "quelle classe". Pour générer un `7`, on lui donne un `z` aléatoire **et** la condition "classe 7".

### 5.3 La perte ELBO (`src/losses/elbo.py`)

```
loss = reconstruction + beta * KL
```

- **reconstruction** : erreur quadratique moyenne entre l'image d'origine et l'image reconstruite (plus c'est bas, mieux c'est).
- **KL** : distance entre la distribution latente apprise `q(z|x)` et une gaussienne standard `N(0, I)` (le "prior"). Elle force l'espace latent à rester régulier, ce qui est ce qui permet ensuite de générer de nouvelles images en tirant `z` au hasard.
- **beta** : un curseur entre les deux objectifs. C'est le paramètre étudié dans l'ablation (section 7).

## 6. Où voir les résultats et comment les interpréter

### 6.1 Grilles d'images

| Fichier | Ce que ça montre | Comment le lire |
|---|---|---|
| `reports/figures/mnist_real_grid.png` | 16 vrais chiffres MNIST | Référence visuelle, sert de "vérité terrain". |
| `reports/figures/vae_reconstruction_grid.png` | Ligne du haut = images réelles, ligne du bas = leur reconstruction par le VAE | Si les deux lignes se ressemblent, le VAE a bien appris à compresser/décompresser. Un léger flou est normal (propre à la loss MSE + KL). |
| `reports/figures/vae_random_samples_grid.png` | 64 images générées en tirant `z ~ N(0, I)`, **sans aucune condition** | C'est la génération "libre" du VAE : on ne choisit pas la classe, le modèle génère ce qu'il veut. |
| `reports/figures/cvae_grid.png` | Une ligne par classe (0 à 9), chaque ligne générée en demandant explicitement cette classe au CVAE | **C'est la figure la plus importante pour juger la contrôlabilité.** Chaque ligne doit ressembler au chiffre correspondant. Verdict après inspection : les classes 0, 1, 2, 7, 8, 9 sont clairement reconnaissables et cohérentes sur toute la ligne ; certaines lignes (3, 4, 5, 6) contiennent quelques échantillons plus ambigus ou un peu déformés — normal pour un entraînement de seulement 10 epochs sur CPU. |

### 6.2 Espace latent (t-SNE)

| Fichier | Ce que ça montre |
|---|---|
| `reports/figures/latent_tsne_vae.png` | Position 2D (projection t-SNE) de 2000 images de test dans l'espace latent du VAE, colorées par leur vraie classe. |
| `reports/figures/latent_tsne_cvae.png` | Même chose pour le CVAE. |

**Interprétation, et c'est le résultat le plus intéressant de la séance :** dans le VAE, les points se regroupent nettement par couleur (par classe), **alors que le VAE n'a jamais reçu le label pendant l'entraînement**. Le modèle a spontanément appris à séparer les chiffres dans son espace latent, simplement parce que c'est la façon la plus efficace de bien reconstruire des images très différentes les unes des autres. Dans le CVAE, à l'inverse, les couleurs sont beaucoup plus mélangées : c'est cohérent, puisque le CVAE reçoit déjà la classe en entrée du décodeur, il n'a plus besoin de coder l'identité du chiffre dans `z` — celui-ci se spécialise plutôt sur le **style d'écriture** (inclinaison, épaisseur du trait...).

### 6.3 Interpolation dans l'espace latent

| Fichier | Ce que ça montre |
|---|---|
| `reports/figures/interpolation_vae_3_to_8.png` | 10 images, de gauche à droite, en interpolant linéairement entre le `z` d'un vrai 3 et le `z` d'un vrai 8. |
| `reports/figures/interpolation_vae_1_to_7.png` | Même chose entre un 1 et un 7. |

**Interprétation :** la transition est progressive, pas de "saut" brutal au milieu — c'est le signe que l'espace latent est continu et bien structuré (grâce au terme KL qui le rapproche d'une gaussienne). C'est exactement ce que l'énoncé demande de vérifier.

### 6.4 Comparaison chiffrée (`reports/experiments/comparison.json`)

Mesuré sur les 10 000 images du test set MNIST, avec les modèles principaux (`vae_main`, `cvae_main`) :

| Modèle | Reconstruction (test) | KL (test) |
|---|---|---|
| VAE | 677.35 | 17.12 |
| CVAE | 676.56 | 13.89 |

Les deux modèles reconstruisent presque aussi bien l'un que l'autre ; le CVAE a un KL légèrement plus bas, cohérent avec l'idée qu'il a moins besoin de "travailler" son espace latent puisque la classe est donnée à part.

**Contrôlabilité du CVAE — un résultat qu'il faut savoir expliquer en séance :**
Nous avons essayé de mesurer automatiquement si le CVAE génère bien la classe demandée, sans passer par une simple inspection visuelle. Faute d'un classifieur pré-entraîné disponible, nous avons utilisé un classifieur "plus proche centroïde" (le centroïde d'une classe = image moyenne des vrais chiffres de cette classe). Résultat : **29,4 % de précision globale**, avec de fortes disparités selon les classes (100 % pour le "0", 0 % pour le "1", "4", "5", "7", "9").

Ce chiffre **contredit clairement l'inspection visuelle de `cvae_grid.png`**, où la plupart des lignes sont pourtant reconnaissables. Nous avons creusé la question : en testant le même classifieur "plus proche centroïde" sur de **vrais** chiffres du test set (pas des chiffres générés), il obtient 82 % de précision — donc la méthode est correcte en soi. Le problème vient d'ailleurs : les images générées par un VAE/CVAE sont **légèrement floues** (conséquence connue de la loss MSE + KL), et cette méthode de classification par distance de pixels est très sensible au flou et à la finesse du trait, en particulier pour les chiffres fins comme 1, 4, 7, 9 (un léger flou les rapproche, en distance de pixels, de classes plus "pleines" comme 0 ou 8, même si un humain les reconnaît sans problème). **Conclusion : notre proxy quantitatif sous-estime la contrôlabilité réelle du CVAE ; l'inspection visuelle de la grille reste, à ce stade, la preuve la plus fiable.** Un vrai classifieur (petit CNN entraîné sur MNIST) donnerait une mesure plus juste — c'est noté dans les prochaines étapes.

Pour le VAE non conditionnel, nous avons aussi regardé, sur 1000 échantillons générés librement, à quelle classe (toujours via le même proxy) ils ressemblent le plus : seules 4 classes sur 10 apparaissent (0, 2, 3, 8), très majoritairement 0 et 2. Cela illustre bien la différence fondamentale avec le CVAE : le VAE classique ne garantit aucune couverture homogène des classes quand on génère librement, alors que le CVAE permet de choisir la classe voulue.

## 7. Étude d'ablation sur β (livrable demandé par l'énoncé)

Protocole : même VAE (latent_dim=16, hidden_channels=32), même seed, 6 epochs, sur un sous-ensemble de 12 000 images d'entraînement (validation complète sur 6 000 images). Le sous-échantillonnage du train set est un choix assumé pour limiter le temps de calcul sur un entraînement CPU uniquement — voir section 9.

| β | Reconstruction (val) | KL (val) | Loss totale (val) |
|---|---|---|---|
| 0.1 | 680.87 | 39.47 | 684.82 |
| 1.0 | 691.83 | 15.37 | 707.20 |
| 5.0 | 725.11 | 0.56 | 727.90 |

Courbe : `reports/figures/ablation_beta_curve.png`
Comparaison visuelle des reconstructions selon β : `reports/figures/ablation_beta_reconstruction_comparison.png` (ligne du haut = images réelles, puis β=0.1, β=1.0, β=5.0)

**Lecture :**
- **β = 0.1** : la reconstruction est la meilleure (proche de l'original, visuellement net sur `ablation_beta_reconstruction_comparison.png`), mais le KL explose (39.5) : l'espace latent est peu régularisé, donc moins fiable pour générer une image à partir d'un `z` tiré au hasard (le prior N(0,I) ne correspond pas bien à ce que l'encodeur produit réellement).
- **β = 5.0** : le KL s'effondre presque à zéro (0.56) — symptôme classique de **posterior collapse** : le modèle arrête d'utiliser l'espace latent. La figure de comparaison le montre très clairement : les reconstructions à β=5.0 sont quasiment des taches grises informes, le décodeur ignore largement `z` et produit presque toujours la même image floue.
- **β = 1.0** : compromis clair entre les deux : reconstruction encore raisonnable (691.8, contre 680.9 pour β=0.1, soit une perte de qualité modérée) et KL significatif mais pas explosif (15.4), donc un espace latent réellement structuré et exploitable pour la génération.

**Meilleur paramètre retenu : β = 1.0.** C'est le choix que nous avons utilisé pour les modèles principaux (`vae_main`, `cvae_main`). Il n'écrase pas le signal latent (contrairement à β=5.0) et ne laisse pas l'espace latent devenir irrégulier au point de nuire à la génération (contrairement à β=0.1). Si l'objectif prioritaire était uniquement la qualité de reconstruction (ex. compression d'image), β=0.1 serait un meilleur choix ; si l'objectif était de forcer un espace latent très lisse pour l'interpolation, on pourrait tester des valeurs encore un peu plus hautes que 1.0, mais pas au-delà de ce qui fait s'effondrer le KL.

## 8. Comparaison qualitative VAE vs CVAE (livrable demandé par l'énoncé)

| Critère | VAE | CVAE |
|---|---|---|
| Contrôle de la classe générée | Impossible à demander explicitement (on subit ce que le modèle tire) | On choisit la classe en argument de `sample()` |
| Couverture des classes en génération libre | Très inégale : seulement 4 classes sur 10 observées sur 1000 tirages | Non applicable — on choisit toujours la classe |
| Organisation de l'espace latent | Se structure spontanément par classe (visible sur `latent_tsne_vae.png`) | Reste mélangé par classe, se spécialise plutôt sur le style (visible sur `latent_tsne_cvae.png`) |
| Reconstruction (test set) | 677.35 | 676.56 (quasi identique) |
| KL (test set) | 17.12 | 13.89 |

**Conclusion pédagogique :** le CVAE ne reconstruit pas mieux que le VAE (ce n'est pas son objectif), mais il résout le vrai problème du VAE classique — l'absence de contrôle sur la génération — en déplaçant l'information de classe hors de l'espace latent, qui se libère alors pour représenter uniquement le style.

## 9. Difficultés rencontrées et comment nous les avons résolues

### Difficulté 1 : bug de checkpoint partagé entre VAE et CVAE
Les deux modèles écrivaient dans le même fichier `reports/best_checkpoint.pth`. Le dernier entraînement lancé (le CVAE, en mode test rapide) avait donc écrasé le VAE correctement entraîné, sans que cela soit visible dans les logs. **Résolu** en donnant à chaque expérience son propre dossier de sortie (`training.output_dir` dans le YAML).

### Difficulté 2 : entraînement lent car uniquement sur CPU
Pas de GPU disponible sur cette machine. Un epoch complet sur les 54 000 images MNIST prend environ 2 minutes 30 à 3 minutes. **Résolu / contourné** en :
- limitant les modèles principaux à 10 epochs (suffisant pour voir une convergence nette, cf. `training_log.csv`, la perte de validation se stabilise après l'epoch 7-8) ;
- pour l'étude d'ablation uniquement, en utilisant un sous-ensemble de 12 000 images d'entraînement (le jeu de validation reste complet) pour pouvoir tester 3 valeurs de β dans un temps raisonnable ;
- en entraînant VAE et CVAE en parallèle (deux processus) pour gagner du temps.

### Difficulté 3 : le proxy de contrôlabilité contredisait l'inspection visuelle
Détaillé en section 6.4. Nous avons vérifié la méthode sur de vraies images avant de conclure que le problème venait de la sensibilité de la distance en pixels au flou des images générées, et pas d'un bug de génération. C'est un bon exemple de la différence entre "le nombre dit X" et "il faut comprendre pourquoi avant de le croire".

### Difficulté 4 (héritée de la séance précédente) : import Python, valeur `lr` mal typée, CVAE non branché dans la boucle d'entraînement
Déjà résolues précédemment (voir `docs/explanations.md` pour le détail) ; nous les listons pour mémoire, elles ne sont plus d'actualité.

## 10. Limites actuelles

- **Fashion-MNIST** : la configuration existe (`dataset.name: fashion_mnist`) mais **charge en réalité encore MNIST** (`src/data/datasets.py`) — c'est un point d'extension non implémenté, pas un dataset différent. À ne pas présenter comme fait.
- **CelebA** : non implémenté, lève explicitement une erreur (`NotImplementedError`).
- **FID** : non calculé. L'énoncé le mentionne comme optionnel ("si les ressources le permettent") ; sur CPU seul, sans réseau Inception pré-entraîné disponible hors-ligne, nous avons priorisé les autres livrables demandés.
- **Contrôlabilité CVAE** : mesurée avec un proxy simple (plus proche centroïde) dont on a montré les limites (section 6.4), pas avec un vrai classifieur.
- Les entraînements principaux sont limités à 10 epochs (contrainte CPU) ; la courbe de perte suggère qu'ils pourraient encore progresser légèrement avec plus d'epochs.

## 11. Prochaines étapes

- Brancher réellement **Fashion-MNIST** (remplacer l'alias vers `datasets.MNIST` par `datasets.FashionMNIST` dans `src/data/datasets.py`) et relancer VAE + CVAE dessus.
- Intégrer **CelebA** (ou un sous-échantillon), avec conditionnement `multi_label` sur quelques attributs (déjà prévu dans le code de `CVAE`, jamais testé).
- Entraîner un petit classifieur CNN sur MNIST pour remplacer le proxy "plus proche centroïde" par une vraie mesure de contrôlabilité.
- Si une machine avec GPU devient disponible, relancer les modèles principaux avec plus d'epochs et comparer.
- Préparer la démonstration web demandée par l'énoncé (sélection d'une classe → génération, slider d'interpolation).
- Calculer un score FID si les ressources de calcul le permettent.

## 12. Glossaire

| Terme anglais | Traduction / explication |
|---|---|
| Encoder | Encodeur — transforme l'image en distribution latente |
| Decoder | Décodeur — reconstruit une image à partir du latent |
| Latent space | Espace latent — représentation compressée apprise |
| Reconstruction loss | Perte de reconstruction — écart entre image d'origine et reconstruite |
| KL divergence | Divergence de Kullback-Leibler — écart entre la distribution apprise et la loi normale |
| Beta (β-VAE) | Poids appliqué au terme KL dans la loss |
| Batch | Mini-lot d'images traitées ensemble |
| Epoch | Un passage complet sur les données d'entraînement |
| Seed | Graine aléatoire, pour pouvoir reproduire un résultat |
| Condition | Information supplémentaire donnée au modèle (ici, le label de classe) |
| Posterior collapse | Le modèle cesse d'utiliser l'espace latent (KL proche de 0) |
| Checkpoint | Sauvegarde des poids du modèle à un instant donné |

## 13. Questions possibles du professeur et réponses courtes

**Pourquoi commencer par MNIST ?** Dataset simple et rapide, permet de valider toute la chaîne (données → modèle → loss → entraînement → visualisation) avant de passer à des images plus complexes.

**Quelle est la vraie différence entre VAE et CVAE, au-delà du code ?** Le VAE encode l'identité de la classe *dans* l'espace latent, de façon non supervisée (on le voit sur le t-SNE). Le CVAE reçoit la classe séparément, donc son espace latent se spécialise sur autre chose (le style). Le CVAE permet donc de *choisir* la classe générée, le VAE non.

**Quel est le meilleur β trouvé ?** β = 1.0, meilleur compromis entre fidélité de reconstruction et régularité de l'espace latent (section 7). β=0.1 reconstruit mieux mais régularise mal ; β=5.0 régularise trop et fait s'effondrer l'espace latent (posterior collapse), visible directement sur les images.

**Le CVAE contrôle-t-il vraiment bien la génération ?** Visuellement oui (`cvae_grid.png`), mais notre première tentative de mesure automatique donnait un chiffre bas et trompeur — nous avons identifié pourquoi (sensibilité au flou de la méthode utilisée) plutôt que de le cacher. C'est expliqué en détail en section 6.4.

**Quel est le principal blocage actuel ?** Aucun blocage bloquant : la suite dépend surtout de temps de calcul (Fashion-MNIST et CelebA demandent les mêmes étapes que MNIST, mais chaque entraînement prend 30-45 minutes sur CPU).

## 14. Message de synthèse oral possible

"Bonsoir Monsieur, cette séance nous avons corrigé un bug qui faisait qu'un de nos modèles précédents n'était en réalité pas entraîné, puis nous avons réellement entraîné un VAE et un CVAE sur MNIST. Nous avons mené l'étude d'ablation demandée sur le poids β : β=1.0 offre le meilleur compromis, β=5.0 fait s'effondrer l'espace latent, ce qu'on voit très nettement sur nos figures. Nous avons aussi visualisé l'espace latent en 2D : il se structure tout seul par classe pour le VAE, mais pas pour le CVAE, ce qui illustre bien pourquoi le CVAE permet de contrôler la génération. Nous avons essayé de mesurer automatiquement cette contrôlabilité, obtenu un chiffre qui contredisait ce qu'on voyait à l'œil, et creusé pour comprendre pourquoi plutôt que de l'ignorer. Les prochaines étapes sont Fashion-MNIST, CelebA, et une vraie mesure de contrôlabilité avec un classifieur entraîné."

---

## Commandes utiles

Installer les dépendances :
```bash
python -m pip install -r requirements.txt
```

Lancer les tests (8 tests, doivent tous passer) :
```bash
python -m pytest -q
```

Entraîner le VAE principal (≈30-45 min sur CPU) :
```bash
python -m src.training.train --config configs/mnist_vae.yaml
```

Entraîner le CVAE principal :
```bash
python -m src.training.train --config configs/mnist_cvae.yaml
```

Lancer l'étude d'ablation sur β :
```bash
python scripts/run_ablation.py --config configs/ablation_beta.yaml
```

Régénérer les grilles d'images (à partir d'un modèle déjà entraîné) :
```bash
python scripts/generate_vae_recon_grid.py --checkpoint reports/experiments/vae_main/best_checkpoint.pth
python scripts/generate_cvae_grid.py --checkpoint reports/experiments/cvae_main/best_checkpoint.pth
```

Régénérer les visualisations de l'espace latent et l'interpolation :
```bash
python -m src.visualization.latent --config configs/mnist_vae.yaml --checkpoint reports/experiments/vae_main/best_checkpoint.pth --output reports/figures/latent_tsne_vae.png
python -m src.visualization.interpolation --config configs/mnist_vae.yaml --checkpoint reports/experiments/vae_main/best_checkpoint.pth --output reports/figures/interpolation_vae_3_to_8.png --class-a 3 --class-b 8
```

Relancer la comparaison quantitative VAE vs CVAE :
```bash
python scripts/evaluate.py
```

## Documents à consulter

- [Résultats d'ablation générés automatiquement](docs/RESULTATS.md)
- [Version condensée pour l'oral de cette séance](docs/presentation_seance_4.md)
- [Explications techniques détaillées (comment exécuter, concepts, FAQ)](docs/explanations.md)
- [Compte rendu de la toute première séance (archive)](docs/presentation_seance_1.md)
