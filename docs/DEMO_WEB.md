# Démonstration web — architecture et mode d'emploi

Ce document couvre la couche de service (API + interface) construite au-dessus
des modèles entraînés par l'équipe. **Aucun code ML n'a été modifié** : ni
`src/` (MNIST, Sylvain), ni `projects/david_fashion_mnist/` (David), ni
`projects/blaise_celeba/` (Blaise). La démo les importe tels quels.

---

## 1. En bref

| | |
|---|---|
| Backend | FastAPI + uvicorn, `backend/` |
| Frontend | React 18 + Vite 6 + Tailwind CSS 4, `frontend/` |
| Inférence | **Exclusivement via le MLflow Model Registry** |
| Datasets | MNIST, Fashion-MNIST, CelebA — sélecteur global |
| Packaging | Un seul container Docker, un seul port |
| Latence décodage | ~34 ms par image (mesuré, CPU) |
| Documentation d'API | auto-générée sur `/docs` |

---

## 2. Deux problèmes, une architecture en deux couches

### Couche 1 — Réconcilier trois implémentations incompatibles

Le dépôt contient **trois** implémentations indépendantes de VAE/CVAE, écrites
séparément. Elles divergent sur presque tout :

| | `src/` (MNIST) | `david_fashion_mnist/` | `blaise_celeba/` |
|---|---|---|---|
| Images | 28×28 gris | 28×28 gris | **64×64 RGB** |
| Normalisation | `[-1, 1]` | `[0, 1]` | `[-1, 1]` |
| Sortie décodeur | Tanh | Sigmoid | Tanh |
| Loss | MSE | BCE | MSE |
| Conditionnement | canaux spatiaux | concat one-hot | **multi-label (3 attributs)** |
| `forward()` | 3 valeurs | 4 valeurs | 3 valeurs |
| Configuration | YAML externe | dans le checkpoint | dans le checkpoint |

Le piège principal est la **plage de sortie**. Une image produite dans `[0, 1]`
mais interprétée comme du `[-1, 1]` donne une image délavée **sans lever la
moindre erreur**.

Réponse : `backend/adapters/`, un protocole `ModelAdapter` et une
implémentation par famille. La plage est une propriété de l'adaptateur, jamais
une constante globale. Un test le garantit
(`test_chaque_famille_declare_sa_propre_plage`), vérifié en sabotant
volontairement la valeur pour confirmer qu'il échoue bien.

CelebA a mis l'abstraction à l'épreuve : Blaise a surchargé `prepare_input`
(la base supposait une image à un seul canal) et traduit ses 3 attributs non
exclusifs en 2³ = 8 combinaisons, plutôt que de modifier l'interface commune
au risque de casser les deux autres familles.

### Couche 2 — Tout faire passer par MLflow

Le déploiement MLflow demandé par l'enseignant n'est pas une brique à côté :
**c'est la seule voie d'inférence de l'application.** Aucune route n'importe
ni n'instancie un modèle ; chacune charge un `mlflow.pyfunc` depuis le Model
Registry.

```
Route FastAPI
   └─ RegistryGateway.load("mnist/cvae_main")
        └─ mlflow.pyfunc.load_model("models:/mnist-cvae_main/1")
             └─ AdapterPyfuncModel  →  ModelAdapter  →  modèle PyTorch
```

Même les métadonnées viennent de MLflow : `latent_dim`, `num_conditions` et
`beta` sont lus dans les paramètres du run d'empaquetage, pas sur un
adaptateur chargé.

---

## 3. Les décisions et leurs raisons

**Une enveloppe pyfunc générique plutôt que celle de David.** Ses wrappers
(`projects/david_fashion_mnist/deployment/`) sont spécifiques à Fashion-MNIST
et n'exposent que `decode`. Reconstruction et interpolation depuis de vraies
images exigent `encode`. `AdapterPyfuncModel` enveloppe la couche
d'adaptateurs et couvre donc les trois familles d'un coup, sans toucher à son
code — qui reste son livrable, avec son propre store MLflow.

**Le Registry plutôt que `mlflow models serve`.** Un serveur MLflow ne sert
qu'un modèle. L'application en utilise sept : suivre sa recette à la lettre
imposerait sept processus le jour de la démonstration. Le chargement depuis le
Registry conserve le versionnage et l'empaquetage reproductible en un seul
processus. Le contrat des modèles restant sérialisable en JSON, les mêmes
modèles sont servables en HTTP sans adaptation.

**FastAPI plutôt que Flask.** L'argument initial contre FastAPI — l'async
n'apporte rien à de l'inférence synchrone en processus — ne tient plus dès
lors que le backend parle à MLflow. S'y ajoutent la validation Pydantic, qui
remplace la validation manuelle, et `/docs` auto-généré.

**Signature MLflow déclarée explicitement.** Inférée depuis un exemple, elle
typait `n` et `seed` en entiers requis alors qu'ils sont facultatifs, et
MLflow rejetait toute requête les laissant à `null`. Les colonnes numériques
facultatives sont donc des `double`, seul type numérique capable de porter une
valeur manquante.

**Code embarqué mis en scène.** `code_paths` copie l'intégralité des dossiers
qu'on lui passe, et `projects/` pèse plus de 100 Mo. Seuls les modules
réellement importés au chargement sont recopiés dans un dossier temporaire,
soit ~470 Ko par modèle.

**Préchauffage en arrière-plan.** La première initialisation MLflow coûte une
trentaine de secondes. Dans le `lifespan`, elle bloquait l'acceptation des
connexions et le serveur paraissait mort ; elle tourne donc dans un thread.

---

## 4. Démarrage

```bash
make install-demo    # dépendances API (torch CPU, mlflow, fastapi) + npm
make fixtures        # échantillons d'images réelles
make register        # empaquette et enregistre les modèles dans MLflow
```

`make register` est **obligatoire** : sans registre, aucune inférence n'est
possible. À relancer après tout ajout de checkpoint.

### Développement

```bash
make dev-api     # terminal 1 — uvicorn avec rechargement à chaud
make dev-web     # terminal 2 — Vite sur :5173, proxifie /api vers :8000
```

### Démonstration

```bash
make demo            # fixtures + register + build + serveur sur :8000
make docker-demo     # variante dockerisée
```

### Inspecter le registre

```bash
make mlflow-ui       # onglet Models de l'interface MLflow
```

---

## 5. Ajouter des checkpoints

Les poids ne sont pas versionnés (sauf ceux de MNIST et les deux
Fashion-MNIST finaux). Emplacements attendus :

| Dataset | Emplacement |
|---|---|
| Fashion-MNIST | `projects/david_fashion_mnist/checkpoints/*.pt` |
| CelebA | `projects/blaise_celeba/results/experiments/<run>/best_checkpoint.pth` |

Déposer les fichiers, puis relancer `make register`. Rien d'autre : le type de
modèle et β sont déduits du nom de fichier ou de la configuration embarquée, et
l'architecture des formes du `state_dict`.

État actuel : MNIST 5 modèles, Fashion-MNIST 2 (seul β = 1 a été transmis,
l'onglet Ablation est donc inactif sur ce dataset), CelebA 0 — le code
d'intégration de Blaise est prêt mais ses checkpoints n'ont pas encore été
déposés.

---

## 6. Les cinq écrans

Un sélecteur global bascule entre les datasets ; les cinq onglets suivent. Les
noms de classes viennent de l'API — chiffres pour MNIST, libellés pour
Fashion-MNIST, combinaisons d'attributs pour CelebA. Rien n'est codé en dur
côté React.

| Onglet | Ce qu'il montre |
|---|---|
| **Génération** | Tirage z ~ N(0, I) puis décodage, classe sélectionnable sur le CVAE. |
| **Interpolation** | Deux images réelles encodées vers mu, interpolation linéaire, décodage de chaque point. |
| **Espace latent** | Un curseur par dimension de z, redécodage à chaque mouvement. |
| **Reconstruction** | Même image reconstruite par le VAE et le CVAE, plus les métriques du dataset. |
| **Ablation β** | Un même z décodé par toute la série de β. |

---

## 7. API

Sous `/api`. Documentation interactive complète sur **`/docs`**. Les
identifiants sont namespacés par dataset (`mnist/cvae_main`).

| Route | Description |
|---|---|
| `GET /api/health` | État, mode d'inférence, modèles au registre |
| `GET /api/datasets` | Datasets disponibles et leurs `class_names` |
| `GET /api/models?dataset=…` | Modèles, avec leur nom et version au registre |
| `GET /api/metrics?dataset=…` | Résultats d'évaluation du dataset |
| `GET /api/fixtures?dataset=…` | Vignettes des images réelles |
| `POST /api/sample` | `{model_id, n, class_label?, seed?}` |
| `POST /api/decode` | `{model_id, z, class_label?}` |
| `POST /api/encode` | `{model_id, index}` → `{z}` |
| `POST /api/reconstruct` | `{model_id, index}` |
| `POST /api/interpolate` | `{model_id, source_index, target_index, steps}` |
| `POST /api/ablation/compare` | `{dataset, series?, z?, seed?}` |

**Codes d'erreur.** 422 quand Pydantic rejette le corps de la requête, en
amont de la route. 400 pour les règles qui dépendent du Registry (classe hors
bornes, dimension de z incorrecte) — Pydantic ne peut pas les connaître. 404
pour un modèle, dataset ou série inconnus. 503 pour un checkpoint, fixture ou
registre absent.

### Contrat du modèle MLflow

Identique en appel direct et en HTTP :

```
colonnes : op (sample|decode|encode), z, class_label, image_base64, n, seed
sortie   : [{images_base64, z, model_id, latent_dim, ...}]
```

---

## 8. Ajouter une famille de modèles

1. écrire `backend/adapters/<nom>.py` (classe héritant de `ModelAdapter` +
   fonction `load`) ;
2. l'enregistrer dans `LOADERS` de `backend/adapters/__init__.py` ;
3. ajouter le `DatasetEntry` dans `build_datasets()` de `backend/catalog.py` ;
4. ajouter le dataset à `scripts/build_demo_fixtures.py` ;
5. `make register`.

Le frontend n'a **pas** à être modifié : il découvre tout via `/api/datasets`
et `/api/models`. Blaise a suivi exactement cette procédure pour CelebA.

---

## 9. Tests

```bash
python -m pytest -q
```

49 tests, paramétrés sur les datasets et ignorés proprement quand les
checkpoints ou le registre correspondants sont absents.

`pytest.ini` restreint la découverte à `tests/` : MLflow embarque une copie du
code source dans chaque modèle packagé, si bien que `deployment/test_serving.py`
existe en plusieurs exemplaires portant le même nom de module. Ces tests de
serving supposent de toute façon des serveurs déjà lancés et se déclenchent à
la main.

---

## 10. Limite connue héritée du modèle MNIST

Le fond des images MNIST générées est **gris et non noir**. Dans
`src/models/vae.py` et `cvae.py`, le dernier `ConvDecoderBlock` applique un
`BatchNorm + ReLU` avant le `Tanh` final : la sortie est bornée dans `[0, 1)`
alors que les données sont dans `[-1, 1]`. Le décodeur ne peut donc jamais
produire le noir.

Les modèles Fashion-MNIST (Sigmoid) et CelebA n'ont pas ce défaut — basculer
entre les datasets dans l'onglet Reconstruction le montre directement.

Le correctif **invalide les checkpoints MNIST existants** et impose de relancer
les entraînements. La décision revient à l'équipe modèle ; la démo signale la
limite explicitement plutôt que de la masquer.
