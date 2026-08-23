# Comprendre le sous-projet CelebA, de zero

Ce document explique tout le travail fait sur CelebA comme si vous n'aviez
jamais touche a un VAE. Chaque choix technique (parametre, taille, seuil) est
justifie : pas juste "on a mis 64", mais "on a mis 64 parce que...". Objectif :
pouvoir relire ce document dans six mois et tout reexpliquer sans avoir a
redecouvrir le code.

---

## 1. C'est quoi le probleme, en une phrase

On veut apprendre a un ordinateur a "comprendre" des visages au point de
pouvoir en inventer de nouveaux, et si possible choisir a l'avance certains
traits du visage invente (par exemple : "genere-moi un visage souriant").

---

## 2. C'est quoi un VAE, vraiment

### 2.1 L'idee de base : compresser puis reconstruire

Un VAE (Variational Autoencoder, autoencodeur variationnel) est fait de deux
morceaux :

- **L'encodeur** : prend une image et la resume en une poignee de nombres
  (le **vecteur latent**, note `z`). Imaginez que vous devez decrire un
  visage a quelqu'un en 64 mots-cles au lieu de montrer la photo : c'est ce
  que fait l'encodeur.
- **Le decodeur** : prend ces 64 nombres et essaie de repeindre l'image
  d'origine a partir d'eux seuls.

Si l'encodeur et le decodeur font bien leur travail ensemble, alors ces 64
nombres contiennent "l'essentiel" du visage (forme, couleur de peau,
coiffure, expression...). C'est ce couple encodeur/decodeur qu'on entraine.

### 2.2 Pourquoi "variationnel" ? Le petit truc qui change tout

Un autoencodeur classique (non variationnel) associerait a une image UN SEUL
point `z`. Probleme : rien n'oblige les points a etre ranges intelligemment
dans l'espace latent. Deux visages tres differents pourraient se retrouver
juste a cote l'un de l'autre, et l'espace entre deux points connus pourrait
ne rien vouloir dire du tout. Consequence : impossible de "generer" un
nouveau visage en tirant un point au hasard, on tomberait sur du bruit.

Le VAE resout ca en associant a chaque image, non pas un point, mais une
**zone floue** (une petite distribution de probabilite, une gaussienne)
decrite par deux vecteurs :

- `mu` (mu) : le centre de la zone (equivalent du point du VAE classique).
- `logvar` : la taille de la zone (plus precisement, le logarithme de sa
  variance ; on travaille en log pour des raisons numeriques, ca evite des
  valeurs negatives impossibles pour une variance).

Le point `z` reellement utilise est ensuite tire au hasard DANS cette zone :

```
z = mu + eps * exp(0.5 * logvar)
```

ou `eps` est un bruit aleatoire standard (`eps ~ N(0, 1)`). C'est **l'astuce
de reparametrisation** : au lieu de tirer `z` directement au hasard (ce qui
empecherait le retropropagation du gradient, l'algorithme qui permet au
reseau d'apprendre), on isole le hasard dans `eps` et on ne fait que des
calculs "normaux" (addition, multiplication) sur `mu` et `logvar`. Le reseau
peut donc apprendre a ajuster `mu` et `logvar` par descente de gradient
comme n'importe quel autre parametre.

### 2.3 La loss ELBO : le "contrat" que le reseau doit respecter

Le reseau est entraine a minimiser une quantite qui a deux morceaux :

```
loss = reconstruction + beta * KL
```

**Le terme reconstruction.** Ecart entre l'image d'origine et l'image
reconstruite par le decodeur (erreur quadratique moyenne, MSE : on met au
carre la difference de chaque pixel, et on additionne). Plus bas = mieux :
le decodeur a bien reconstruit l'image a partir des 64 nombres.

**Le terme KL (divergence de Kullback-Leibler).** Mesure a quel point la
zone floue `(mu, logvar)` de chaque image ressemble a une gaussienne
standard bien "propre" `N(0, I)` (centree en zero, variance 1). Ce terme
force TOUTES les zones floues a rester proches les unes des autres et
proches du centre. C'est ce qui rend l'espace latent utilisable pour
generer : si on tire un point au hasard dans `N(0, I)`, il tombera dans une
zone que le decodeur a deja appris a reconstruire.

**Le compromis.** Ces deux objectifs se tirent dans des directions
opposees : minimiser uniquement la reconstruction pousserait chaque image
vers une zone floue minuscule et tres specifique (bonne reconstruction,
mais espace latent desordonne) ; minimiser uniquement la KL pousserait
toutes les zones floues vers exactement la meme gaussienne (espace latent
tres propre, mais le decodeur ne pourrait plus distinguer les images entre
elles). `beta` est le curseur qui pese ce compromis, voir section 7.

### 2.4 Et le CVAE (Conditional VAE) dans tout ca ?

Le CVAE est un VAE auquel on donne une information supplementaire, la
**condition**, a la fois a l'encodeur et au decodeur. Dans notre cas, la
condition est un petit vecteur qui dit "ce visage sourit / ne sourit pas",
"c'est un homme / une femme", etc. (section 5).

Grace a ca, le decodeur recoit toujours deux informations : "quelle forme
dans l'espace latent" (`z`) et "quels attributs" (la condition). Pour
generer un visage souriant, on lui donne un `z` au hasard **et** la
condition "Smiling=1".

---

## 3. Pourquoi CelebA est plus difficile que MNIST

| | MNIST (Sylvain) | CelebA (ce sous-projet) |
|---|---|---|
| Contenu | Chiffres manuscrits 0-9 | Visages de celebrites |
| Taille image | 28x28 pixels | 64x64 pixels |
| Couleur | Niveaux de gris (1 canal) | Couleur (3 canaux, rouge/vert/bleu) |
| "Classe" | Un chiffre exclusif (0 OU 1 OU ... OU 9) | Des attributs qui peuvent se cumuler (sourire ET homme ET...) |
| Quantite d'info par image | Faible (une forme simple) | Grande (texture de peau, cheveux, fond, eclairage...) |

Consequence directe sur l'architecture (section 4) et sur la facon de
conditionner le CVAE (section 5) : on ne peut pas juste recopier ce qui a
ete fait pour MNIST, il a fallu adapter.

---

## 4. L'architecture du reseau, bloc par bloc

### 4.1 Pourquoi 4 etages et pas 2 comme pour MNIST

Le reseau MNIST compresse l'image en 2 etapes (2 convolutions). Ici, on en
utilise 4. Raison : une image MNIST 28x28 en niveaux de gris est simple (un
trait blanc sur fond noir) ; une image CelebA 64x64 en couleur contient
beaucoup plus de details, il faut donc plus d'etapes de compression pour
resumer toute cette information dans un vecteur de taille raisonnable.

```
Image d'entree : 3 x 64 x 64  (3 canaux couleur, 64 pixels de large, 64 de haut)
     |  Conv2d 3 -> 32 canaux, on divise la taille par 2
     v
32 x 32 x 32
     |  Conv2d 32 -> 64 canaux, on divise la taille par 2
     v
64 x 16 x 16
     |  Conv2d 64 -> 128 canaux, on divise la taille par 2
     v
128 x 8 x 8
     |  Conv2d 128 -> 256 canaux, on divise la taille par 2
     v
256 x 4 x 4
     |  on "aplatit" tout en une seule liste de nombres
     v
4096 nombres (256 x 4 x 4)
     |  deux petites couches separees
     v
mu (64 nombres)      logvar (64 nombres)
```

**Pourquoi les canaux doublent a chaque etage (32, 64, 128, 256) ?**
Convention tres courante en vision par ordinateur : a chaque fois qu'on
divise la taille spatiale de l'image par 2 (on perd du detail geometrique),
on double le nombre de "filtres" (canaux) pour compenser en gardant la
capacite du reseau a representer des motifs varies (couleur, texture, etc).

**Pourquoi `latent_dim = 64` (64 nombres pour resumer un visage) ?** Chez
Sylvain, MNIST utilise 16 nombres. Un visage porte beaucoup plus
d'information qu'un chiffre (identite, pose, expression, style de cheveux,
eclairage...), on lui donne donc plus de "place" pour se resumer. 64 est un
choix raisonnable et courant dans la litterature pour des visages a cette
resolution, pas une valeur magique : plus grand donnerait probablement une
meilleure reconstruction mais un espace latent plus dur a regulariser
(rappel section 2.3) ; plus petit forcerait une compression plus agressive
et donc plus de perte de detail.

**Chaque bloc de convolution contient aussi :**
- `BatchNorm2d` : normalise les valeurs a l'interieur du reseau pendant
  l'entrainement, ce qui le rend plus stable et plus rapide a entrainer.
- `LeakyReLU(0.2)` au lieu d'un `ReLU` classique cote encodeur : un `ReLU`
  classique met a zero toutes les valeurs negatives, ce qui peut, avec un
  reseau profond comme celui-ci (4 etages), faire "mourir" certains
  neurones (ils ne s'activent plus jamais et n'apprennent plus rien).
  `LeakyReLU` laisse passer un tout petit peu de signal cote negatif
  (multiplie par 0.2 au lieu de 0), ce qui reduit ce risque.

### 4.2 Le decodeur : l'inverse exact

Le decodeur fait le chemin inverse avec des `ConvTranspose2d` (l'operation
inverse d'une convolution qui reduit la taille : celle-ci l'augmente) :
`4096 -> 4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x32 -> 64x64x3`, jusqu'a
retrouver une image de la meme taille que celle de depart.

**Detail important, corrige par rapport a l'implementation MNIST de
l'equipe :** la toute derniere etape du decodeur n'a NI `BatchNorm` ni
`ReLU`, juste la convolution suivie d'un `Tanh` (une fonction qui ecrase
n'importe quelle valeur dans l'intervalle `[-1, 1]`). Chez Sylvain, la
derniere etape garde un `ReLU` avant le `Tanh`, ce qui a pour consequence
que le noir pur (`-1`) ne peut jamais etre reellement atteint (documente
dans `backend/adapters/sylvain_mnist.py`). Ici, en separant la derniere
etape des autres, on evite cette limite : le `Tanh` peut vraiment produire
toute la plage `[-1, 1]`.

### 4.3 Et le CVAE, concretement, qu'est-ce qui change ?

Le CVAE a exactement la meme structure convolutive. La seule difference :
apres avoir aplati l'image en 4096 nombres, on **colle** (concatene) le
vecteur de condition (3 nombres, un par attribut) a la suite, ce qui donne
4099 nombres au lieu de 4096, avant de calculer `mu` et `logvar`. Meme chose
cote decodeur : on colle la condition a `z` avant de le redeployer en image.

**Pourquoi coller la condition apres l'aplatissement, et pas au tout debut
comme chez Sylvain (qui ajoute des "canaux" supplementaires directement sur
l'image) ?** Deux raisons :
1. Un attribut de visage (sourire, genre) concerne l'image ENTIERE, ce
   n'est pas localise a un endroit precis comme peut l'etre la forme d'un
   chiffre. Coller l'information a un vecteur deja "global" (apres
   l'aplatissement) colle mieux a cette idee.
2. C'est moins cher en calcul : etaler 3 canaux constants sur une image de
   64x64 et les faire traverser 4 convolutions coute plus cher que
   d'ajouter 3 nombres a une simple couche de connexion.

---

## 5. Les donnees : d'ou elles viennent et comment on les choisit

### 5.1 Le probleme du telechargement officiel

CelebA se telecharge normalement via `torchvision.datasets.CelebA`, qui
pointe vers un lien Google Drive partage. Ce lien tombe tres souvent en
erreur de quota depasse, un probleme connu et documente qui touche beaucoup
de monde, pas seulement cette machine.

**Solution utilisee** : on charge les memes images et les memes attributs
depuis un miroir hebergé sur Hugging Face (`tpremoli/CelebA-attrs`), qui
propose exactement les memes 3 lots officiels (train/validation/test) sans
jamais passer par Google Drive.

### 5.2 Le sous-echantillonnage : pourquoi et comment

L'enonce du sujet demande explicitement d'utiliser un sous-echantillon de
CelebA (pas les 200 000+ images completes). Raison pratique en plus : sans
carte graphique disponible sur cette machine (tout tourne sur le processeur,
le CPU), entrainer sur l'integralite prendrait beaucoup trop de temps.

**Comment le sous-echantillon est tire.** On ne prend pas simplement les N
premieres images du jeu de donnees : CelebA range ses photos par personne
(plusieurs photos consecutives de la meme celebrite), donc prendre un
"prefixe" donnerait un echantillon domine par une poignee de visages. A la
place, on melange aleatoirement tout le jeu de donnees (avec une **seed**,
un nombre de depart fixe pour le generateur aleatoire, ici `42`, ce qui
rend le tirage reproductible : relancer le meme code redonne exactement le
meme echantillon) puis on prend les N premiers apres melange.

**Tailles utilisees, et pourquoi elles different entre les modeles
principaux et l'ablation :**

| Usage | images d'entrainement | validation | test |
|---|---|---|---|
| VAE et CVAE principaux | 8 000 | 1 500 | 1 500 |
| Etude d'ablation (chaque valeur de beta) | 4 000 | 1 000 | 1 000 |

L'ablation entraine 3 modeles complets (un par valeur de beta), donc pour
rester dans un temps de calcul raisonnable sur CPU, elle utilise un
echantillon d'entrainement plus petit. Le jeu de validation/test reste assez
grand (1 000 images) pour donner une mesure fiable malgre tout.

### 5.3 Le choix des 3 attributs de conditionnement

CelebA fournit 40 attributs binaires par image (sourire, lunettes, couleur
de cheveux, etc). On ne peut pas tous les utiliser : avec 40 attributs, il y
aurait 2^40 combinaisons possibles, un nombre astronomique, et un
sous-echantillon de quelques milliers d'images ne pourrait jamais couvrir
qu'une fraction infime de ces combinaisons.

**On a donc mesure, sur un echantillon reel, la frequence de plusieurs
attributs candidats avant de choisir**, plutot que de deviner :

| Attribut | Frequence mesuree |
|---|---|
| Smiling (sourit) | 47,6 % |
| Male (homme) | 42,0 % |
| Wavy_Hair (cheveux ondules) | 30,6 % |
| Eyeglasses (lunettes), candidat ecarte | 7,2 % |
| Wearing_Hat (porte un chapeau), candidat ecarte | 5,2 % |

**Attributs retenus : `Smiling`, `Male`, `Wavy_Hair`.** Deux criteres :

1. **On les voit a l'oeil nu sur une photo**, ce qui permet de verifier
   visuellement si le modele a bien appris a les respecter.
2. **Ils sont assez frequents pour ne pas creer de combinaison presque
   vide.** Avec 3 attributs binaires, il y a 2^3 = 8 combinaisons possibles
   (sourit ou pas, homme ou femme, cheveux ondules ou pas, toutes les
   combinaisons). Un attribut rare comme `Eyeglasses` (7,2 %) aurait cree
   des combinaisons quasiment vides (par exemple "porte des lunettes ET a
   les cheveux ondules" concernerait moins de 2 % des images). Avec les 3
   attributs retenus, meme la combinaison la plus rare represente encore
   2,5 % de l'echantillon, soit environ 200 images sur les 8 000
   d'entrainement : suffisant pour un reseau qui partage ses poids entre
   toutes les combinaisons (ce n'est pas 8 reseaux separes, un seul reseau
   apprend a gerer les 8 cas a la fois).

---

## 6. Le protocole d'entrainement, parametre par parametre

| Parametre | Valeur | Pourquoi |
|---|---|---|
| `seed` | 42 | Nombre de depart du generateur aleatoire. Fixe pour que deux lancements du meme code donnent exactement le meme resultat (meme tirage des donnees, meme initialisation du reseau). |
| Optimiseur | Adam | Algorithme standard pour entrainer des reseaux de neurones, ajuste automatiquement la vitesse d'apprentissage pour chaque parametre du reseau. Choix par defaut tres repandu, pas de raison specifique de s'en ecarter ici. |
| `lr` (taux d'apprentissage) | 0.001 | Controle la taille des pas que le reseau fait a chaque mise a jour de ses poids. Valeur par defaut classique pour Adam, ni trop grande (risque d'instabilite) ni trop petite (apprentissage trop lent). |
| `batch_size` (taille de lot) | 64 | Nombre d'images regardees en meme temps avant chaque mise a jour du reseau. 64 est un compromis courant entre vitesse (plus grand = moins de mises a jour) et memoire disponible (plus grand = plus de memoire necessaire), adapte a un entrainement CPU. |
| `epochs` (modeles principaux) | 18 | Un "epoch" = un passage complet sur toutes les images d'entrainement. 18 a ete choisi en observant que la perte de validation continue de baisser jusque-la sans signe evident de sur-apprentissage (voir `results/experiments/*/training_log.csv`). |
| `epochs` (ablation) | 10 | Reduit par rapport aux modeles principaux pour que les 3 entrainements de l'etude d'ablation restent faisables en temps raisonnable sur CPU. |
| `beta` (modeles principaux) | 1.0 | Valeur retenue apres l'etude d'ablation (section 7) comme meilleur compromis. |

---

## 7. L'etude d'ablation sur beta : ce qu'elle montre et pourquoi c'est demande

"Ablation" = on fait varier un seul reglage (ici, `beta`) en gardant tout le
reste identique (meme seed, meme architecture, meme nombre d'epochs, meme
sous-echantillon), pour isoler precisement l'effet de ce reglage.

| beta | reconstruction (validation) | KL (validation) |
|---|---|---|
| 0.1 | 605.00 (la meilleure) | 279.01 (la plus haute) |
| 1.0 | 656.34 | 124.01 |
| 5.0 | 821.32 (la pire) | 56.86 (la plus basse) |

**Comment lire ce tableau.** Rappel section 2.3 : `beta` pese le compromis
entre bien reconstruire et avoir un espace latent bien range. Le tableau
confirme exactement ce compromis :
- Avec un `beta` petit (0.1), le reseau se concentre presque uniquement sur
  la reconstruction (elle est la meilleure des 3), au prix d'un espace
  latent peu discipline (KL la plus haute).
- Avec un `beta` grand (5.0), c'est l'inverse : l'espace latent est tres
  discipline (KL la plus basse) mais la reconstruction en souffre nettement
  (821, contre 605 pour beta=0.1, soit environ 36 % de perte de qualite).
- `beta = 1.0` est le compromis retenu pour les modeles principaux : ni
  l'un ni l'autre effet n'est extreme.

---

## 8. Comment lire les figures produites

### 8.1 Grilles d'images

- `vae_reconstruction_grid.png` : ligne du haut = vrais visages du jeu de
  test, ligne du bas = leur reconstruction par le VAE. Un leger flou est
  normal (consequence connue de la loss MSE + KL, le meme flou existe sur
  les figures MNIST et Fashion-MNIST du reste de l'equipe).
- `vae_random_samples_grid.png` : visages **inventes** en tirant `z` au
  hasard dans `N(0, I)`, sans aucune condition. C'est la generation
  "libre" du VAE, on ne choisit pas les traits du visage genere.
- `cvae_grid.png` : une ligne par combinaison des 3 attributs (8 lignes au
  total). Chaque ligne est generee en demandant explicitement cette
  combinaison au CVAE : c'est la figure la plus importante pour juger si le
  CVAE respecte bien ce qu'on lui demande.

### 8.2 Espace latent en 2D (t-SNE)

`latent_tsne_vae.png` et `latent_tsne_cvae.png` : le t-SNE est une methode
qui prend les 64 nombres du vecteur latent de chaque image et les projette
sur seulement 2 axes, pour pouvoir les dessiner sur un graphique. Chaque
point colore represente une image du jeu de test.

**Une difference notable avec MNIST, a bien comprendre.** Chez Sylvain, les
points du VAE MNIST se regroupent nettement par chiffre sur ce genre de
graphique, alors que le chiffre n'est jamais donne au VAE pendant
l'entrainement. Ici, les points ne se regroupent PAS nettement selon
l'attribut `Smiling` : bleu (ne sourit pas) et rouge (sourit) restent
melanges partout sur le graphique. Ce n'est pas un signe d'echec :
l'explication la plus probable est qu'un chiffre manuscrit est domine par sa
forme (un 3 et un 8 sont tres differents visuellement), alors qu'un visage
varie surtout selon la pose, l'identite ou l'eclairage, des facteurs qui
pesent plus lourd, en distance de pixels, qu'un simple sourire. Le VAE,
entraine uniquement a bien reconstruire, n'a donc aucune raison particuliere
de faire du sourire un axe majeur de son espace latent.

### 8.3 Interpolation

`interpolation_vae_0_to_1.png` : on prend deux vrais visages du jeu de
test, on calcule leurs deux vecteurs latents (`mu`), puis on fabrique 10
points intermediaires en interpolant lineairement entre les deux, et on
demande au decodeur de dessiner chacun de ces points. Si la transition est
progressive (pas de "saut" brutal au milieu), c'est le signe que l'espace
latent est continu et bien structure, exactement ce que le terme KL est
cense garantir (section 2.3).

---

## 9. L'evaluation chiffree

### 9.1 Reconstruction et KL sur le jeu de test

| Modele | Reconstruction (test) | KL (test) |
|---|---|---|
| VAE | 500.75 | 129.67 |
| CVAE | 479.19 | 119.36 |

Le CVAE reconstruit legerement mieux que le VAE. Explication : le CVAE
recoit deja les attributs en entree du decodeur, il n'a donc pas besoin
d'utiliser une partie de son espace latent pour "deviner" ces attributs, qui
peut alors se concentrer sur le reste (style, identite), un peu plus facile
a reconstruire une fois les attributs fixes.

### 9.2 Est-ce que le CVAE respecte vraiment les attributs demandes ?

Question difficile a mesurer automatiquement sans un vrai classifieur
entraine (pas disponible ici). On utilise donc un **proxy simple** : pour
chaque attribut, on calcule l'image moyenne de tous les vrais visages
d'entrainement ou l'attribut est actif, et l'image moyenne de tous ceux ou
il ne l'est pas (deux "centroides"). Une image generee est jugee "correcte"
si elle est plus proche, en distance de pixels, du bon centroide.

| Attribut | Precision de ce proxy |
|---|---|
| Smiling | 60,1 % |
| Male | 51,9 % |
| Wavy_Hair | 56,3 % |
| Moyenne | 56,1 % |

Ces chiffres depassent le hasard (50 %, puisque chaque attribut est binaire)
mais restent modestes, surtout pour `Male`. **A prendre avec prudence** :
une image generee par un VAE est naturellement floue, et une methode qui
compare des distances de pixels est tres sensible a ce flou, elle peut donc
sous-estimer la vraie qualite du controle exerce par le CVAE. L'inspection a
l'oeil de `cvae_grid.png` suggere une meilleure controlabilite que ce chiffre
seul ne le laisse penser. Un vrai classifieur entraine donnerait une mesure
plus fiable (piste listee dans les prochaines etapes).

**Le score FID (souvent utilise pour juger la qualite d'images generees)
n'a pas ete calcule** : l'enonce le presente comme optionnel ("si les
ressources le permettent"), et il necessiterait un reseau de classification
d'images pre-entraine (Inception) que cette machine, sans connexion
garantie et sans GPU, ne peut pas telecharger et faire tourner dans un temps
raisonnable.

---

## 10. Comment tout ca s'assemble dans la demo web de l'equipe

Le depot contient une petite application web (backend Flask + frontend
React) qui permet de manipuler les modeles sans toucher au code. Elle sert
deja les modeles MNIST (Sylvain) et Fashion-MNIST (David), chacun via un
**adaptateur** : une petite couche qui traduit les particularites de chaque
modele (plage de valeurs des pixels, facon de donner la condition, etc.)
vers une interface commune, pour que le reste de l'application n'ait jamais
besoin de savoir quel modele il utilise reellement.

CelebA suit exactement le meme principe (`backend/adapters/blaise_celeba.py`) :

- Le VAE/CVAE CelebA produit des images dans `[-1, 1]` (a cause du `Tanh`
  final, section 4.2), l'adaptateur le declare explicitement pour que
  l'affichage final soit correct.
- Comme les attributs CelebA ne sont pas exclusifs (contrairement a un
  chiffre 0-9), mais que l'application ne sait parler que d'un entier
  "classe" unique, l'adaptateur traduit un entier de 0 a 7 vers l'une des
  8 combinaisons possibles des 3 attributs.
- Les checkpoints (fichiers de poids entraines) sont decouverts
  automatiquement sur le disque plutot que listes a la main : plus simple
  a maintenir, et coherent avec la facon dont les checkpoints de David sont
  geres.

---

## 11. Glossaire rapide

| Terme | Explication |
|---|---|
| Encodeur | Partie du reseau qui transforme une image en vecteur latent. |
| Decodeur | Partie du reseau qui reconstruit une image a partir d'un vecteur latent. |
| Espace latent | L'ensemble de tous les vecteurs `z` possibles ; une representation compressee et apprise des images. |
| Reparametrisation | L'astuce mathematique qui permet d'entrainer un reseau malgre un tirage aleatoire (section 2.2). |
| Perte de reconstruction | Ecart mesure entre l'image d'origine et l'image reconstruite. |
| Divergence KL | Mesure d'ecart entre la distribution apprise et une gaussienne standard. |
| Beta (beta-VAE) | Poids applique au terme KL dans la loss, curseur du compromis reconstruction/regularite. |
| Condition | Information supplementaire donnee au CVAE en plus du vecteur latent (ici, les 3 attributs). |
| Batch (lot) | Groupe d'images traitees ensemble avant une mise a jour du reseau. |
| Epoch | Un passage complet sur toutes les images d'entrainement. |
| Seed | Nombre de depart d'un generateur aleatoire, fixe pour rendre un resultat reproductible. |
| Checkpoint | Fichier qui sauvegarde l'etat (les poids) d'un reseau entraine, pour pouvoir le recharger plus tard sans le reentrainer. |
| Sur-apprentissage (overfitting) | Quand un modele "apprend par coeur" les donnees d'entrainement et devient moins bon sur des donnees jamais vues. |

---

## 12. Pour aller plus loin dans le code

- `data/dataset.py` : chargement et sous-echantillonnage des donnees (section 5).
- `models/layers.py`, `models/vae.py`, `models/cvae.py` : l'architecture (section 4).
- `losses/elbo.py` : la loss ELBO (section 2.3).
- `training/trainer.py`, `training/train.py` : la boucle d'entrainement (section 6).
- `evaluation/run_ablation.py` : l'etude d'ablation (section 7).
- `evaluation/generate_grids.py`, `evaluation/latent_visualization.py`, `evaluation/interpolation.py` : les figures (section 8).
- `evaluation/evaluate.py` : l'evaluation chiffree (section 9).
- `backend/adapters/blaise_celeba.py` (a la racine du depot) : l'integration a la demo (section 10).

Pour les commandes exactes permettant de relancer chaque etape, voir la
section "Commandes utiles" de `README.md` dans ce meme dossier.
