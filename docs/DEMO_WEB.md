# Démonstration web — architecture et mode d'emploi

Ce document couvre la couche de service (API + interface) construite au-dessus
des modèles entraînés par l'équipe. **Aucun code ML n'a été modifié** : ni
`src/` (MNIST, Sylvain), ni `projects/david_fashion_mnist/` (Fashion-MNIST,
David). La démo les importe tels quels.

---

## 1. En bref

| | |
|---|---|
| Backend | Flask 3 + waitress, `backend/` |
| Frontend | React 18 + Vite 6, `frontend/` |
| Inférence | PyTorch CPU, checkpoints des deux sous-projets |
| Datasets | MNIST et Fashion-MNIST, sélecteur global |
| Packaging | Un seul container Docker, un seul port |
| Latence décodage | ~14 ms par image (mesuré, CPU) |

---

## 2. Le problème central : deux modèles incompatibles

Le dépôt contient **deux implémentations indépendantes** de VAE/CVAE, écrites
séparément. Elles divergent sur presque tout :

| | `src/` (MNIST) | `projects/david_fashion_mnist/` |
|---|---|---|
| Normalisation des données | `[-1, 1]` | `[0, 1]` |
| Sortie du décodeur | Tanh | **Sigmoid** |
| Loss de reconstruction | MSE | BCE |
| BatchNorm | oui | non |
| Couche FC cachée | non | oui (`hidden_dim=256`) |
| Conditionnement du CVAE | canaux spatiaux | concat one-hot après flatten |
| Retour de `forward()` | 3 valeurs | 4 valeurs |
| Signature de `sample()` | `sample(c, n)` | `sample(labels)` |
| Configuration | YAML externe | dans le checkpoint |
| Paramètres | ~257 k | ~1,69 M |
| β de l'ablation | 0.1 / 1.0 / 5.0 | 0.1 / 1 / 4 |

Le piège principal est la **plage de sortie**. Une image produite dans `[0, 1]`
mais interprétée comme du `[-1, 1]` donne une image délavée **sans lever la
moindre erreur** — un bug qui ne se voit qu'à l'œil, et seulement si on sait
quoi chercher.

### La réponse : une couche d'adaptateurs

Réécrire l'une des deux implémentations pour la faire ressembler à l'autre
aurait invalidé des checkpoints déjà entraînés et détruit le travail d'un
membre de l'équipe. À la place, chaque famille est enveloppée :

```
backend/adapters/
  base.py            # protocole ModelAdapter (abstrait)
  sylvain_mnist.py   # enveloppe src/models/*
  david_fashion.py   # enveloppe projects/david_fashion_mnist/models/*
```

Le protocole expose `encode`, `decode`, `latent_dim`, `is_conditional`,
`input_range` et `output_range`. **Aucune route de l'API ne manipule un modèle
directement.** La plage de valeurs est une propriété de l'adaptateur, jamais
une constante globale — c'est ce qui empêche le bug ci-dessus.

Un test le garantit explicitement
(`test_chaque_famille_declare_sa_propre_plage`) ; il a été vérifié en
sabotant volontairement la valeur pour confirmer qu'il échoue bien.

---

## 3. Autres choix techniques

**Flask plutôt que FastAPI.** L'atout principal de FastAPI est l'async, or
l'inférence PyTorch est synchrone et CPU-bound : l'async n'apporte rien ici. On
perd la validation Pydantic, remplacée par une validation explicite dans
`backend/app.py` et `ModelAdapter.validate_latent` — sans elle, un payload
malformé remonterait en 500 depuis torch au lieu d'un 400 lisible.

**Un seul container.** Le frontend est buildé puis servi par Flask sur le même
port. Pas de docker-compose, pas de CORS en production, une seule commande à
lancer le jour de la présentation.

**Images en 28×28 natif.** Les décodeurs produisent du 28×28 ; on l'envoie tel
quel (~200 octets par image en PNG) et c'est le CSS qui agrandit avec
`image-rendering: pixelated`. Envoyer des images upscalées côté serveur
coûterait 50× plus de bande passante pour zéro information supplémentaire.

**Fixtures d'images précalculés.** Reconstruction et interpolation ont besoin
de vraies images. Plutôt que d'embarquer les datasets (~64 Mo chacun),
`scripts/build_demo_fixtures.py` extrait 12 images par classe et par dataset
dans des `.npz` de 20 à 51 Ko. Ils sont stockés en **uint8 brut** : la
normalisation est appliquée par l'adaptateur cible, puisque les deux familles
n'attendent pas la même.

**Découverte des modèles.** MNIST utilise une liste explicite (checkpoints
versionnés, chemins stables). Fashion-MNIST utilise un **glob** sur
`projects/david_fashion_mnist/checkpoints/*.pt` : ces poids sont gitignorés et
arrivent hors du dépôt, avec des noms de run qui dépendent des arguments
d'entraînement. Figer une liste obligerait à la corriger à chaque fichier reçu.

**Taille de l'image Docker : ~1,33 Go.** L'essentiel vient de PyTorch, qui pèse
754 Mo décompressé sur Linux même en build CPU. Le Dockerfile retire
`torch/include` et `torch/test` (~180 Mo). L'élagage s'arrête là
volontairement : `torch/bin` (`torch_shm_manager`), `torch/lib` et `torchgen`
sont tous chargés à l'import, et les supprimer fait échouer le démarrage.

---

## 4. Intégrer les checkpoints Fashion-MNIST

Les poids de David **ne sont pas versionnés** : son `.gitignore` exclut `*.pt`,
`*.pth` et `checkpoints/`. Ils se déposent manuellement dans
`projects/david_fashion_mnist/checkpoints/`, et sont détectés au démarrage.

### État de la livraison

Deux checkpoints ont été transmis, décrits par lui comme « sélectionnés pour
l'application web » :

| Fichier | Type | β | Epoch | Taille |
|---|---|---|---|---|
| `vae_beta_1_seed42_final.pt` | VAE | 1.0 | 92 | 19,3 Mo |
| `cvae_beta_1_seed42_final.pt` | CVAE | 1.0 | 92 | 19,4 Mo |

Intégrité vérifiée contre les SHA256 de son `model_manifest.json`.

**Les runs β = 0.1 et β = 4 n'ont pas été transmis.** L'onglet Ablation est donc
inactif sur Fashion-MNIST : une série d'un seul β ne permet aucune comparaison.
L'écran l'explique et renvoie vers le tableau de chiffres, qui reste
consultable puisqu'il vient de ses CSV. Les quatre autres onglets fonctionnent
normalement. Ajouter les fichiers manquants suffira à activer l'écran, sans
aucune modification de code.

### Ce qui est automatique

- le **type** (VAE ou CVAE) et le **β** sont lus dans le nom du fichier —
  convention de David, où le point décimal est supprimé (`beta_01` = 0.1). Les
  suffixes de run (`_seed42_final`) sont ignorés ;
- l'**architecture** (`latent_dim`, `hidden_dim`, `num_classes`) est déduite
  des formes du `state_dict`. Les poids sont la seule source de vérité, donc un
  checkpoint aux métadonnées incohérentes se chargera quand même.

### Chargement

`torch.load` désérialise par défaut via pickle, ce qui **exécute du code
contenu dans le fichier**. Les checkpoints étant reçus de l'extérieur, ils sont
lus avec `weights_only=True` : ils ne contiennent que des tenseurs et des types
primitifs, cette restriction suffit donc. Un repli vers le chargement complet
existe pour d'éventuels checkpoints plus anciens, mais n'est pas atteint par
les poids actuels.

En Docker, `make docker-run-mounted` monte le dossier depuis l'hôte, ce qui
évite de reconstruire l'image à chaque changement de poids.

---

## 5. Démarrage

### Installation (une fois)

```bash
make install-demo
make fixtures
```

### Développement (rechargement à chaud)

Deux terminaux :

```bash
make dev-api
```

```bash
make dev-web
```

Frontend sur http://localhost:5173, API sur http://localhost:8000. Vite
proxifie `/api` vers Flask, donc le code frontend n'utilise que des chemins
relatifs et reste identique en dev et en démo.

### Démonstration

```bash
make demo
```

Builde le frontend et le sert depuis Flask sur http://localhost:8000.
Variante dockerisée : `make docker-demo`.

---

## 6. Les cinq écrans

Un sélecteur global en en-tête bascule entre MNIST et Fashion-MNIST ; les cinq
onglets suivent. Les noms de classes viennent de l'API : chiffres pour MNIST,
libellés (« Basket », « Sac »…) pour Fashion-MNIST — rien n'est codé en dur
côté React.

| Onglet | Ce qu'il montre |
|---|---|
| **Génération** | Tirage z ~ N(0, I) puis décodage. Sur le CVAE, la classe est sélectionnable. |
| **Interpolation** | Deux images réelles encodées vers mu, interpolation linéaire, décodage de chaque point. |
| **Espace latent** | Un curseur par dimension de z, redécodage à chaque mouvement. Bouton « partir d'une image réelle ». |
| **Reconstruction** | Même image reconstruite par le VAE et le CVAE, plus les métriques du dataset actif. |
| **Ablation β** | Un même z décodé par toute la série de β. Sur Fashion-MNIST, deux séries au choix (VAE / CVAE). |

L'onglet Ablation est celui qui apporte le plus par rapport aux figures
statiques : il rend l'effondrement du postérieur visible et manipulable.
Comparer les deux datasets sur cet onglet est parlant — le KL chute de 39 à
0,56 sur MNIST contre 41 à 6,8 sur Fashion-MNIST, plus texturé.

---

## 7. API

Toutes les routes sont sous `/api`. Les images sont des data-URI PNG base64.
Les identifiants de modèles sont **namespacés par dataset** (`mnist/cvae_main`,
`fashion/cvae_beta_1`).

### Métadonnées

| Route | Description |
|---|---|
| `GET /api/health` | État, device, version torch, modèles disponibles |
| `GET /api/datasets` | Datasets ayant au moins un checkpoint, avec leurs `class_names` |
| `GET /api/models?dataset=…` | Modèles du dataset |
| `GET /api/metrics?dataset=…` | JSON de `reports/` (MNIST) ou CSV de David converti (Fashion) |
| `GET /api/fixtures?dataset=…` | Vignettes des images réelles, groupées par classe |

### Génération

| Route | Corps | Réponse |
|---|---|---|
| `POST /api/sample` | `{model_id, n, class_label?, seed?}` | `{images: [...]}` |
| `POST /api/decode` | `{model_id, z, class_label?}` | `{image}` |
| `POST /api/encode` | `{model_id, index}` | `{z, true_label}` |
| `POST /api/reconstruct` | `{model_id, index}` | `{original, reconstruction}` |
| `POST /api/interpolate` | `{model_id, source_index, target_index, steps}` | `{images, alphas, source, target}` |
| `POST /api/ablation/compare` | `{dataset, series?, z?, seed?, class_label?}` | `{results, z, series, available_series}` |

`class_label` est obligatoire pour les modèles conditionnels et ignoré sinon.
`seed` rend un tirage reproductible via un `torch.Generator` dédié, sans
toucher au RNG global.

Sur `/api/ablation/compare`, `series` sélectionne le groupe comparé (`vae` ou
`cvae`). Les modèles sont regroupés par série parce que comparer un VAE et un
CVAE à β différent mélangerait deux effets distincts.

### Erreurs

Toutes les erreurs renvoient `{"error": "message en clair"}` avec un statut
approprié : 400 (payload invalide), 404 (modèle, dataset ou série inconnus),
503 (checkpoint ou fixture absent).

---

## 8. Ajouter un dataset ou un modèle

**Nouveau checkpoint dans une famille existante** — pour MNIST, ajouter une
entrée dans `_mnist_models()` de `backend/catalog.py`. Pour Fashion-MNIST,
simplement déposer le `.pt` : il est découvert automatiquement.

**Nouvelle famille de modèles** (CelebA, ou une autre implémentation) :

1. écrire `backend/adapters/<nom>.py` avec une classe héritant de
   `ModelAdapter` et une fonction `load(checkpoint_path, device, **kwargs)` ;
2. l'enregistrer dans `LOADERS` de `backend/adapters/__init__.py` ;
3. ajouter le `DatasetEntry` correspondant dans `build_datasets()` ;
4. ajouter le dataset à `scripts/build_demo_fixtures.py`.

Le frontend n'a **pas** à être modifié : il découvre tout via `/api/datasets`
et `/api/models`.

---

## 9. Tests

```bash
python -m pytest tests/test_api.py -q
```

41 tests, paramétrés sur les deux datasets et ignorés proprement quand les
checkpoints correspondants sont absents. Ils couvrent les métadonnées, le
namespacing des identifiants, les plages de valeurs par famille, le
déterminisme (même seed → mêmes images), l'effet du conditionnement (même z +
classe différente → image différente), la cohérence de l'interpolation aux
bornes, la complétude de la série d'ablation, et 12 cas de payloads invalides.

Suite complète : 49 tests.

---

## 10. Limite connue héritée du modèle MNIST

Le fond des images MNIST générées est **gris et non noir**. Ce n'est pas un
problème d'affichage : dans `src/models/vae.py` et `src/models/cvae.py`, le
dernier `ConvDecoderBlock` applique un `BatchNorm + ReLU` avant le `Tanh`
final. La sortie est donc bornée dans `[0, 1)` alors que les données sont
normalisées dans `[-1, 1]`.

Conséquence : le décodeur ne peut jamais produire la valeur `-1` du fond noir.
Cela gonfle la perte de reconstruction (~677) et se voit sur toutes les figures
du dépôt, pas seulement dans la démo.

Les modèles Fashion-MNIST, qui se terminent par une Sigmoid cohérente avec des
données en `[0, 1]`, n'ont pas ce défaut — basculer entre les deux datasets
dans l'onglet Reconstruction le montre directement.

Le correctif tient en quelques lignes côté modèle, mais il **invalide les
checkpoints MNIST existants** et impose de relancer les entraînements. La
décision revient à l'équipe modèle ; la démo signale la limite explicitement
plutôt que de la masquer.
