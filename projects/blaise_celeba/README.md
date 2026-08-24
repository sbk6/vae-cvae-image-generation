# VAE et CVAE sur CelebA

## Contribution CelebA, Blaise

Ce sous-projet etudie le VAE (Variational Autoencoder) et le CVAE
(Conditional VAE) sur CelebA, un jeu de photos de visages de celebrites avec
40 attributs binaires par image (sourire, genre, couleur de cheveux, port de
lunettes, etc.). C'est le troisieme dataset de l'equipe, aux cotes de MNIST
(`src/`, chiffres manuscrits) et Fashion-MNIST (`projects/david_fashion_mnist/`,
vetements). Chaque implementation reste independante, un choix explique dans
`docs/DEMO_WEB.md` : reecrire l'une pour ressembler a une autre invaliderait
des checkpoints deja entraines et melangerait le travail des trois membres.

Le travail realise comprend :

- l'implementation d'un VAE convolutionnel pour des visages couleur 64x64 ;
- l'implementation d'un CVAE conditionne par un vecteur multi-hot de 3
  attributs (contrairement a une classe exclusive comme un chiffre) ;
- une strategie de chargement de CelebA qui evite le probleme recurrent de
  quota Google Drive de `torchvision.datasets.CelebA` ;
- l'entrainement des deux modeles sur un sous-echantillon reproductible et
  equilibre par combinaison d'attributs ;
- une etude d'ablation sur le poids beta, d'abord baseline puis grille fine ;
- une evaluation quantitative (reconstruction, KL, controlabilite du CVAE) ;
- la visualisation de l'espace latent (t-SNE) et l'interpolation ;
- l'integration a la demo web partagee de l'equipe (`backend/`, `frontend/`).

---

## 1. Jeu de donnees

### 1.1 Pourquoi ne pas utiliser `torchvision.datasets.CelebA`

CelebA officiel (mmlab.ie.cuhk.edu.hk/projects/CelebA.html) se telecharge par
defaut via `torchvision.datasets.CelebA(download=True)`, qui pointe vers un
lien Google Drive partage. Ce lien renvoie tres frequemment une erreur de
quota depasse ("Google Drive - Quota exceeded") au lieu du fichier attendu.
Ce n'est pas un probleme propre a cette machine : c'est un incident recurrent
et documente, y compris dans les issues du depot officiel de torchvision et
sur le forum PyTorch.

**Solution retenue** : charger CelebA depuis le miroir Hugging Face
`tpremoli/CelebA-attrs`, qui republie les memes 203 000 images avec les
memes 40 attributs binaires et les memes splits officiels train (162 770),
validation (19 962) et test (19 867), au format parquet, sans jamais passer
par Google Drive. Les 42 colonnes exposees ont ete verifiees directement
(`datasets.load_dataset_builder`) avant d'ecrire le code de chargement,
plutot que supposees identiques a la documentation CelebA officielle.

### 1.2 Sous-echantillonnage equilibre

L'enonce demande explicitement un sous-echantillon de CelebA. Le baseline
utilisait un melange reproductible du split complet, puis selectionnait un
prefixe. La configuration amelioree va plus loin : elle construit un
sous-echantillon equilibre par combinaison des 3 attributs de conditionnement
(`Smiling`, `Male`, `Wavy_Hair`). Concretement, les images sont regroupees
selon les 8 vecteurs possibles, puis on prend autant que possible le meme
nombre d'images dans chaque groupe (voir `data/dataset.py`).

Ce choix est plus logique pour un CVAE : le modele doit apprendre toutes les
conditions, pas seulement les combinaisons naturellement majoritaires dans
CelebA.

Tailles utilisees :

| Usage | n_train | n_val | n_test |
|---|---|---|---|
| VAE et CVAE principaux ameliores | 32 000 | 3 000 | 3 000 |
| Etude d'ablation fine (par valeur de beta) | 8 000 | 2 000 | 2 000 |
| Baseline historique | 8 000 | 1 500 | 1 500 |

**Pourquoi pas directement la moitie du dataset ?** La moitie du split train
CelebA ferait environ 81 000 images, mais ce n'est pas automatiquement plus
pertinent ici. Avec 3 attributs binaires, certaines combinaisons sont rares.
Prendre une tres grande portion aleatoire reproduirait surtout le desequilibre
naturel du dataset : les conditions frequentes domineraient encore, et les
conditions rares resteraient plus difficiles a apprendre. `32 000` images est
un compromis plus defendable pour ce projet : 4 fois plus que le baseline,
mais encore compatible avec un equilibrage fort des 8 combinaisons et avec un
entrainement CPU raisonnable. Si une machine GPU est disponible, on peut
monter plus haut, mais il faudra alors accepter un equilibrage moins strict
ou changer les attributs retenus.

Chaque sous-echantillon (images redimensionnees en 64x64, attributs extraits)
est mis en cache localement apres le premier chargement
(`projects/blaise_celeba/data_cache/`, gitignore) pour eviter de retelecharger
ou reredimensionner a chaque lancement de script.

### 1.3 Attributs de conditionnement

CelebA expose 40 attributs binaires. Utiliser les 40 pour le CVAE ferait
exploser le nombre de combinaisons (2^40) sans qu'aucune ne soit representee
plus d'une fois dans un sous-echantillon de quelques milliers d'images :
inexploitable. Un sous-ensemble restreint est donc necessaire, et son choix a
ete verifie empiriquement sur la distribution naturelle du dataset plutot que
suppose (`evaluation/inspect_attributes.py --n-samples 4000`) :

| Attribut | Frequence naturelle mesuree (4 000 images, seed 42) |
|---|---|
| Smiling | 47,6 % |
| Male | 42,0 % |
| Wavy_Hair | 30,6 % |
| Eyeglasses (candidat ecarte) | 7,2 % |
| Wearing_Hat (candidat ecarte) | 5,2 % |

**Attributs retenus : `Smiling`, `Male`, `Wavy_Hair`.** Deux criteres ont
guide ce choix :

1. **Distinctifs visuellement**, pour pouvoir juger a l'oeil sur les figures
   si le CVAE respecte bien la condition demandee (contrairement a un
   attribut subtil comme `Pointy_Nose`).
2. **Suffisamment equilibres.** Avec 3 attributs binaires, le vecteur
   multi-hot du CVAE a 2^3 = 8 combinaisons possibles. Un attribut rare comme
   `Eyeglasses` (7,2 %) donnerait des combinaisons quasi vides (par exemple
   "porte des lunettes ET cheveux ondules" concernerait moins de 2 % des
   images), avec trop peu d'exemples reels pour que le decodeur apprenne
   correctement cette combinaison. Verification sur l'echantillon reel avec
   les 3 attributs retenus : en tirage aleatoire, la combinaison la plus
   rare (`Smiling=1, Male=1, Wavy_Hair=1`) represente encore 2,5 % des
   images, soit environ 200 exemples sur le baseline de 8 000 images.
   Avec le sous-echantillonnage equilibre actuel, chaque combinaison vise
   environ 4 000 exemples dans les 32 000 images d'entrainement,
   suffisant pour un reseau qui partage ses poids entre toutes les
   combinaisons (ce n'est pas un classifieur separe par combinaison).

---

## 2. Modeles

### 2.1 Architecture

Encodeur et decodeur a 4 etages convolutifs (contre 2 pour MNIST/Fashion-MNIST
dans le reste du depot) :

```
Image 3 x 64 x 64
    | Conv 3->hidden_channels, stride 2
    v hidden_channels x 32 x 32
    | Conv hidden_channels->2*hidden_channels, stride 2
    v 2*hidden_channels x 16 x 16
    | Conv 2*hidden_channels->4*hidden_channels, stride 2
    v 4*hidden_channels x 8 x 8
    | Conv 4*hidden_channels->8*hidden_channels, stride 2
    v 8*hidden_channels x 4 x 4
    | Flatten -> 8*hidden_channels*4*4
    v
    fc_mu, fc_logvar -> latent_dim
```

Le decodeur est le miroir exact (ConvTranspose2d), termine par `Tanh` (sortie
dans `[-1, 1]`).

**Pourquoi 4 etages et pas 2 comme pour les chiffres MNIST/Fashion-MNIST ?**
Un visage couleur 64x64 porte beaucoup plus d'information par image qu'un
chiffre 28x28 en niveaux de gris (texture de peau, couleur des cheveux,
arriere-plan, eclairage). Avec seulement 2 etages, la carte de
caracteristiques avant l'aplatissement resterait a 16x16, et le reseau
n'aurait pas assez de profondeur pour apprendre des motifs visuels aussi
varies. Le baseline historique utilisait `hidden_channels=32` et
`latent_dim=64`. La configuration amelioree utilise maintenant
`hidden_channels=64` et `latent_dim=128`, pour donner plus de capacite au
modele sur des visages couleur ; l'arret anticipe limite le risque de
sur-apprentissage et evite de payer inutilement toutes les epochs si la
validation stagne.

**Pourquoi LeakyReLU plutot que ReLU dans l'encodeur ?** Avec un reseau plus
profond (4 etages), le risque d'unites mortes (neurones qui cessent de
s'activer) augmente. `LeakyReLU(0.2)` laisse passer un gradient faible mais
non nul cote negatif, ce qui limite ce risque.

**Pourquoi le dernier etage du decodeur n'a ni BatchNorm ni ReLU avant le
Tanh ?** Correction assumee d'une limite constatee dans l'implementation
MNIST de l'equipe : son dernier bloc de decodage applique un ReLU avant le
Tanh final, ce qui rend la moitie basse de la plage de sortie `[-1, 1]`
inatteignable en pratique (documente dans `backend/adapters/sylvain_mnist.py`).
En separant le dernier etage du reste du decodeur (voir
`models/layers.py`), ce modele n'a pas cette limite : la plage de sortie
nominale du Tanh est reellement atteignable.

### 2.2 VAE (`models/vae.py`)

Reference : Kingma & Welling, *Auto-Encoding Variational Bayes* (2013).
Encodeur -> `mu`, `logvar` -> reparametrisation (`z = mu + eps * exp(0.5 *
logvar)`, `eps ~ N(0, I)`) -> decodeur. Identique dans son principe a
l'implementation MNIST de l'equipe.

### 2.3 CVAE (`models/cvae.py`)

Reference : Sohn, Lee & Yan, *Learning Structured Output Representation
using Deep Conditional Generative Models* (2015).

**Difference de nature du conditionnement par rapport a MNIST/Fashion-MNIST.**
Un chiffre est une classe exclusive : un exemple est soit un 3, soit un 8,
jamais les deux. Un visage n'a pas cette contrainte : il peut etre a la fois
`Smiling` et `Male`. Le CVAE de ce sous-projet recoit donc un vecteur
multi-hot de dimension 3 (une composante 0/1 par attribut, independamment
des autres), et non un one-hot exclusif.

**Comment la condition est injectee.** Contrairement au CVAE MNIST de
l'equipe, qui diffuse la condition comme des canaux spatiaux constants
concatenes a l'image des l'entree de l'encodeur, ce CVAE concatene le vecteur
de condition au vecteur de caracteristiques *apres* l'aplatissement de la
derniere carte de convolution, cote encodeur comme cote decodeur. Deux
raisons a ce choix :

1. Un attribut de visage (sourire, genre, texture de cheveux) est une
   propriete globale de l'image entiere, pas un signal localise dans une
   region precise de l'image, contrairement a la forme d'un chiffre. Le
   concatener a un vecteur de caracteristiques deja global colle mieux a
   cette semantique.
2. Diffuser 3 canaux constants sur une image de 64x64 et les faire traverser
   4 convolutions coute nettement plus cher en calcul, sur CPU, qu'ajouter
   3 colonnes a une couche entierement connectee.

### 2.4 Perte ELBO (`losses/elbo.py`)

```
loss = reconstruction + beta * KL
```

**Reconstruction : MSE, pas BCE.** L'implementation Fashion-MNIST de
l'equipe utilise la BCE (entropie croisee binaire), qui correspond a un
decodeur de Bernoulli dans la derivation de l'ELBO : hypothese adaptee a des
pixels quasi binaires (fond noir, trait blanc). Ce sous-projet utilise la
MSE (erreur quadratique), qui correspond a un decodeur gaussien : hypothese
plus adaptee a des intensites de pixel continues, comme une photo couleur.
C'est une divergence assumee entre les 3 sous-projets, pas un oubli.

**KL** : forme analytique fermee entre `q(z|x) = N(mu, diag(exp(logvar)))`
et le prior `N(0, I)`, comme les deux autres implementations de l'equipe.

**Beta** : poids du terme KL (beta-VAE, Higgins et al. 2017), etudie en
section 4.

Convention de normalisation (somme sur les pixels puis moyenne sur le batch)
identique a celle de `src/losses/elbo.py` (Sylvain), pour que les valeurs de
reconstruction/KL des deux sous-projets restent comparables dans le rapport
final malgre la difference de taille d'image.

---

## 3. Protocole experimental

- Seed fixe : 42, pour le tirage du sous-echantillon comme pour
  l'initialisation et l'entrainement du modele.
- Optimiseur Adam, taux d'apprentissage 0.001, taille de batch 64.
- Modeles principaux ameliores : `latent_dim=128`, `hidden_channels=64`,
  100 epochs sur 32 000 images equilibrees, avec arret anticipe (`patience=10`,
  `min_delta=1.0`). Le checkpoint `best_checkpoint.pth` garde la meilleure
  validation, et `last_checkpoint.pth` garde le dernier etat atteint.
- `beta=0.5` avec KL annealing lineaire de 0.0 a 0.5 sur les 10 premieres
  epochs. L'objectif est de laisser le decodeur apprendre a reconstruire
  avant de regulariser fortement l'espace latent, puis de rester sous
  `beta=1.0` pour reduire le flou observe dans le baseline.
- Pas de GPU disponible : tous les entrainements tournent sur CPU.

---

## 4. Etude d'ablation sur beta (livrable demande par l'enonce)

Protocole baseline deja execute : meme VAE (`latent_dim=64`,
`hidden_channels=32`), meme seed, 10 epochs, sur le sous-echantillon
d'ablation (4 000 train / 1 000 val / 1 000 test). Memes 3 valeurs que
Sylvain sur MNIST (0.1, 1.0, 5.0), pour permettre une comparaison directe
entre les deux datasets dans le rapport final.

| beta | reconstruction (val) | KL (val) | loss totale (val) |
|---|---|---|---|
| 0.1 | 605.00 | 279.01 | 632.90 |
| 1.0 | 656.34 | 124.01 | 780.35 |
| 5.0 | 821.32 | 56.86 | 1105.63 |

**Lecture :** le compromis attendu s'observe nettement, dans les memes
proportions relatives que sur MNIST. Entre beta=0.1 et beta=5.0, la
reconstruction se degrade d'environ 36 % (605 -> 821) pendant que la KL est
divisee par presque 5 (279 -> 57). Aucun des trois runs ne montre
d'effondrement complet du posterior (KL proche de 0, comme observe par
Sylvain a beta=5.0 sur MNIST) : a beta=5.0 la KL reste a 56.86, un ordre de
grandeur au-dessus de zero. Interpretation : CelebA porte davantage
d'information par image que MNIST (texture, couleur, pose), donc meme
fortement penalise, l'espace latent garde une utilite minimale pour la
reconstruction. Ce premier resultat montre surtout que la zone interessante
est entre `0.1` et `1.0`, car `5.0` degrade trop la reconstruction.

La configuration actuelle lance donc une ablation plus fine :
`[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`, avec 100 epochs par beta sur
8 000 images equilibrees. Les scripts comparent maintenant la
meilleure validation atteinte par chaque run, pas simplement la derniere
epoch, ce qui est plus juste quand les runs peuvent s'arreter a des moments
differents. Le candidat par defaut pour les modeles ameliores est `beta=0.5`
avec KL annealing ; il devra etre confirme par cette ablation fine.

Tableau genere automatiquement : `results/RESULTATS.md`. Courbe : `results/figures/ablation_beta_curve.png`.

---

## 5. Resultats

### 5.1 Grilles d'images qualitatives

| Fichier | Contenu |
|---|---|
| `results/figures/vae_reconstruction_grid.png` | Ligne du haut : visages reels de test. Ligne du bas : leur reconstruction par le VAE. |
| `results/figures/vae_random_samples_grid.png` | Visages generes en tirant `z ~ N(0, I)`, sans condition. |
| `results/figures/cvae_grid.png` | Une ligne par combinaison des 3 attributs (8 lignes), chaque ligne generee en demandant explicitement cette combinaison au CVAE. |

**Lecture :** les reconstructions restent floues, un effet connu et attendu
de la loss MSE + KL (le meme flou est documente sur MNIST et Fashion-MNIST
par le reste de l'equipe), mais conservent la pose, l'orientation du visage
et les grands traits (couleur de cheveux, presence de lunettes) de l'image
d'origine, signe que l'encodeur capture bien l'information utile. Sur la
grille CVAE, l'inspection visuelle des 8 lignes montre des tendances
coherentes avec les attributs demandes (par exemple, les lignes associees a
`Male=1` produisent des visages aux traits plus masculins), mais avec des
exceptions ligne par ligne : attendu avec le baseline de 8 000 images. La
configuration actuelle augmente le train set a 32 000 images equilibrees
pour reduire ce probleme.

### 5.2 Espace latent (t-SNE)

CelebA n'a pas de classes exclusives comme MNIST : il n'existe pas de
"la" classe a colorer sur la projection. La couleur represente donc un seul
attribut a la fois (`Smiling` par defaut, le plus visible sur un visage) ;
voir `evaluation/latent_visualization.py --color-attribute`.

| Fichier | Contenu |
|---|---|
| `results/figures/latent_tsne_vae.png` | Espace latent du VAE, colore par `Smiling`. |
| `results/figures/latent_tsne_cvae.png` | Espace latent du CVAE, colore par `Smiling`. |

**Lecture, et une difference notable avec MNIST.** Chez Sylvain, le VAE MNIST
separe spontanement les points par classe de chiffre dans son espace latent,
alors que le label ne lui est jamais donne. Ici, ni le VAE ni le CVAE ne
montrent de separation nette par `Smiling` : les points bleus et rouges
restent melanges sur toute la projection, pour les deux modeles. Explication
plausible : un chiffre manuscrit est domine par sa forme globale (l'identite
du chiffre est la principale source de variance entre deux images), alors
qu'un visage varie surtout selon des facteurs comme la pose, l'identite, la
couleur de peau et l'eclairage, qui pesent davantage sur la distance en
pixels qu'un sourire. Le VAE, entraine uniquement a bien reconstruire,
n'a donc aucune raison de faire du sourire un axe dominant de son espace
latent. Ce constat n'est pas un signe d'echec du modele : il illustre plutot
que la structure spontanee de l'espace latent depend fortement de ce qui
domine la variance visuelle du dataset, pas seulement de l'architecture.

### 5.3 Interpolation

| Fichier | Contenu |
|---|---|
| `results/figures/interpolation_vae_0_to_1.png` | Interpolation lineaire entre deux visages reels du test set (VAE). |

**Lecture :** la transition entre les deux visages est progressive, sans
saut brutal au milieu de la sequence. Le sourire, l'orientation du visage et
la couleur de peau evoluent continument d'une extremite a l'autre. C'est
exactement ce que l'enonce demande de verifier : un espace latent continu et
bien structure, consequence directe de la regularisation par le terme KL
(section 2.4 et 4).

### 5.4 Comparaison quantitative VAE vs CVAE

Mesure baseline sur les 1 500 images du test set CelebA, modeles historiques
(`vae_main`, `cvae_main`, `latent_dim=64`, `hidden_channels=32`, beta=1.0,
18 epochs) :

| Modele | Reconstruction (test) | KL (test) |
|---|---|---|
| VAE | 500.75 | 129.67 |
| CVAE | 479.19 | 119.36 |

**Lecture :** le CVAE reconstruit legerement mieux que le VAE (479 contre
501) et a une KL plus basse (119 contre 130), le meme sens d'ecart que
Sylvain a observe sur MNIST, avec la meme explication : le CVAE recoit deja
l'attribut en entree du decodeur, il n'a donc pas besoin de coder cette
information dans `z`, qui se libere pour representer le reste (style,
identite), un peu plus facile a reconstruire une fois l'attribut fixe.

**Controlabilite du CVAE**, mesuree avec le meme type de proxy que Sylvain
sur MNIST (plus proche centroide, ici par attribut plutot que par classe,
voir `evaluation/evaluate.py`) : pour chaque attribut, un centroide "actif"
et un centroide "inactif" sont calcules a partir des vraies images
d'entrainement, et chaque image generee est classee en comparant sa distance
aux deux.

| Attribut | Precision (proxy) |
|---|---|
| Smiling | 60,1 % |
| Male | 51,9 % |
| Wavy_Hair | 56,3 % |
| **Moyenne** | **56,1 %** |

Ces chiffres depassent le hasard (50 %) mais restent modestes, en
particulier pour `Male` (51,9 %, a peine mieux qu'un tirage aleatoire). Deux
lectures possibles, pas mutuellement exclusives : (1) le CVAE baseline
controle imparfaitement cet attribut avec seulement 8 000 images
d'entrainement partagees entre 8 combinaisons, ou (2) le proxy lui-meme
sous-estime la
controlabilite reelle, exactement comme Sylvain l'a constate et documente
sur MNIST (le flou des images generees par un VAE degrade la fiabilite d'une
mesure fondee sur la distance en pixels). L'inspection visuelle de
`cvae_grid.png` (section 5.1) suggere une controlabilite superieure a ce que
le chiffre seul indique, dans le meme sens que la conclusion de Sylvain :
**le proxy quantitatif est une mesure de confort, pas une preuve
definitive** ; un vrai classifieur entraine sur CelebA donnerait une mesure
plus fiable (section 7).

FID explicitement hors-perimetre, meme justification que pour MNIST :
optionnel selon l'enonce ("si les ressources le permettent"), pas de reseau
Inception pre-entraine disponible hors ligne, CPU uniquement.

### 5.5 Integration a la demo web partagee

Les checkpoints historiques (`vae_main`, `cvae_main`, `beta_0.1`,
`beta_1.0`, `beta_5.0`) sont decouverts automatiquement par
`backend/catalog.py` et servis par `backend/adapters/blaise_celeba.py`, selon
le meme mecanisme que MNIST et Fashion-MNIST (voir `docs/DEMO_WEB.md`). Les
nouveaux runs (`vae_improved`, `cvae_improved`, ablation fine) suivront le
meme format de checkpoint, donc ils seront deployables de la meme facon une
fois entraines. Verifie de bout en bout sans modifier le frontend (deja
pilote par les donnees du catalogue) :

- `GET /api/datasets` liste bien `celeba` aux cotes de `mnist`, avec les 8
  noms de classes lisibles (une combinaison d'attributs par classe) ;
- `POST /api/sample` sur `celeba/cvae_main` avec `class_label=6` (`Smiling=1`,
  `Male=1`, `Wavy_Hair=0`) renvoie une image PNG
  coherente avec la condition demandee ;
- le chemin reconstruction (`prepare_input` -> `encode` -> `decode` ->
  `to_display_range`) fonctionne avec le fixture RGB
  `backend/assets/celeba_samples.npz`, genere par
  `scripts/build_demo_fixtures.py --dataset celeba` a partir du cache local
  de test (pas de nouveau telechargement).

---

## 6. Limites actuelles

- Sous-echantillon de 32 000 images d'entrainement equilibrees sur les
  162 770 disponibles dans le split train complet : beaucoup plus solide que
  le baseline 8 000, mais encore inferieur au dataset complet.
- 3 attributs de conditionnement sur les 40 disponibles, choisis pour rester
  exploitables avec un sous-echantillon (section 1.3) : pas une couverture
  complete des attributs CelebA.
- Pas de GPU disponible : le protocole ameliore limite l'entrainement a
  100 epochs maximum, mais l'arret anticipe peut couper avant si la
  validation stagne.
- Controlabilite du CVAE mesuree par un proxy "plus proche centroide" par
  attribut (section 5.4), pas par un classifieur entraine : memes limites de
  principe que le proxy utilise par Sylvain sur MNIST (sensible au flou des
  images generees par un VAE).

## 7. Prochaines etapes possibles

- Executer l'ablation beta fine pour confirmer ou remplacer le candidat
  `beta=0.5`.
- Entrainer les configurations ameliorees `vae_improved` et `cvae_improved`,
  puis regenerer les figures et `comparison.json`.
- Sur une machine avec GPU, augmenter encore le sous-echantillon (par exemple
  50 000 images d'entrainement ou plus) pour ameliorer la nettete, en
  verifiant que les 8 combinaisons restent assez representees.
- Etendre le CVAE a davantage d'attributs, avec un sous-echantillon plus
  grand pour compenser la sparsite combinatoire.
- Remplacer le proxy de controlabilite par un petit classifieur CNN
  multi-label entraine sur CelebA.
- Utiliser les runs MLflow maintenant integres pour comparer proprement
  ablations, VAE et CVAE avant le deploiement final.

## 8. Suivi MLflow

MLflow est active par defaut dans les configs (`mlflow.enabled: true`). Les
runs sont stockes localement dans une base SQLite
`projects/blaise_celeba/mlflow.db` via `tracking_uri: sqlite:///mlflow.db`.

Chaque entrainement logge :

- les parametres de config aplatis (`dataset.*`, `model.*`, `training.*`) ;
- les metriques `train_loss`, `val_loss`, reconstruction, KL et beta a chaque
  epoch ;
- `best_val_loss` et `best_epoch` ;
- les artefacts `resolved_config.yaml`, `training_log.csv`,
  `best_checkpoint.pth` et `last_checkpoint.pth`.

Pour ouvrir l'interface MLflow :

```bash
cd projects/blaise_celeba
. .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Puis ouvrir `http://127.0.0.1:5000` dans le navigateur.

Pour desactiver MLflow temporairement, mettre `enabled: false` dans le bloc
`mlflow` de la config concernee.

---

## Commandes utiles

Toutes les commandes ci-dessous supposent d'etre dans
`projects/blaise_celeba/`, environnement virtuel active (`. .venv/bin/activate`).

Installer les dependances :
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Verifier la frequence naturelle des attributs candidats (utile pour justifier
le choix de la section 1.3, mais ce n'est pas le sampling d'entrainement) :
```bash
python -m evaluation.inspect_attributes --n-samples 4000
```

Verifier la distribution reellement utilisee par l'entrainement equilibre :
```bash
python -m evaluation.inspect_attributes --config configs/celeba_vae.yaml --split train
```

Entrainer le VAE principal :
```bash
python -m training.train --config configs/celeba_vae.yaml
```

Entrainer le CVAE principal :
```bash
python -m training.train --config configs/celeba_cvae.yaml
```

Lancer l'etude d'ablation sur beta :
```bash
python -m evaluation.run_ablation --config configs/ablation_beta.yaml
```

Regenerer les grilles qualitatives (a partir d'un modele deja entraine) :
```bash
python -m evaluation.generate_grids --model vae --config configs/celeba_vae.yaml \
    --checkpoint results/experiments/vae_improved/best_checkpoint.pth
python -m evaluation.generate_grids --model cvae --config configs/celeba_cvae.yaml \
    --checkpoint results/experiments/cvae_improved/best_checkpoint.pth
```

Regenerer la visualisation de l'espace latent et l'interpolation :
```bash
python -m evaluation.latent_visualization --config configs/celeba_vae.yaml \
    --checkpoint results/experiments/vae_improved/best_checkpoint.pth \
    --output results/figures/latent_tsne_vae.png
python -m evaluation.interpolation --config configs/celeba_vae.yaml \
    --checkpoint results/experiments/vae_improved/best_checkpoint.pth \
    --output results/figures/interpolation_vae_0_to_1.png
```

Relancer la comparaison quantitative VAE vs CVAE :
```bash
python -m evaluation.evaluate
```

## Documents lies

- [Explications generales du sujet et etat du projet complet](../../docs/state.md)
- [Architecture de la demo web partagee](../../docs/DEMO_WEB.md)
- [Rapport MNIST (Sylvain)](../../docs/presentation_seance_4.md)
- [Rapport Fashion-MNIST (David)](../david_fashion_mnist/README.md)
