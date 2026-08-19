# Déploiement des modèles avec MLflow — guide pour l'équipe web

Ce document explique comment démarrer le service de génération d'images et comment l'appeler depuis un backend (Flask, FastAPI, Node, etc.). Il est destiné à la personne qui développe l'application web de démonstration ; elle n'a besoin de connaître ni PyTorch, ni MLflow, ni l'architecture des modèles.

## 1. Deux fonctionnalités, un seul endpoint, tous les datasets

L'énoncé du projet demande deux choses pour la démo web :
1. **Sélection d'une classe cible → génération d'images** (utilise le **CVAE**, action `generate`).
2. **Slider d'interpolation dans l'espace latent** (utilise le **VAE**, action `interpolate`).

L'équipe travaille en plus sur 3 datasets (MNIST, Fashion-MNIST, CelebA), chacun avec ses propres VAE/CVAE entraînés séparément. **Il n'y a pourtant qu'un seul serveur, un seul port, une seule URL à utiliser** pour tout ça : les champs `action` et `dataset` dans la requête indiquent quoi faire et avec quel modèle. Pas besoin de gérer plusieurs ports ou plusieurs endpoints côté application web.

```
[Frontend]  ->  [Backend web]  ->  HTTP POST /invocations  ->  [Serveur MLflow - un seul port]
                                                                     |
                                                     route en interne vers la bonne
                                                     action ("generate"/"interpolate")
                                                     et le bon dataset
                                                                     |
                                                              image generee
```

À ce jour, seul **MNIST** est disponible pour les deux actions (voir [section 7](#7-datasets-disponibles)).

## 2. Démarrer le serveur

Depuis la racine de ce dépôt (nécessite que `mlflow.db` et le modèle enregistré soient présents — voir `scripts/register_generation_model.py` si besoin de le régénérer) :

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/image_generator/2" -p 5001 --env-manager local
```

- `-m "models:/image_generator/2"` : `2` est le numéro de version actuel (voir [section 7](#7-datasets-disponibles)).
- `-p 5001` : port d'écoute (à adapter si besoin).
- `--env-manager local` : utilise l'environnement Python déjà installé (plus rapide pour le développement local).

Le serveur met quelques dizaines de secondes à démarrer. Il est prêt quand `curl http://127.0.0.1:5001/ping` répond `200 OK`. Les logs de démarrage indiquent aussi la liste des datasets effectivement chargés pour chaque action, par exemple :
```
[generation_pyfunc] datasets disponibles (génération) : ['mnist']
[generation_pyfunc] datasets disponibles (interpolation) : ['mnist']
```

## 3. Endpoints exposés

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/invocations` | POST | Génère ou interpole une image. C'est l'endpoint unique. |
| `/ping` | GET | Vérifie que le serveur est démarré et prêt. |
| `/version` | GET | Version de MLflow utilisée. |

## 4. Action `generate` — sélection d'une classe

### Requête

```json
{
  "dataframe_records": [
    { "action": "generate", "dataset": "mnist", "classe": 7 }
  ]
}
```

- `action` (optionnel, défaut `"generate"`).
- `dataset` (optionnel, défaut `"mnist"`).
- `classe` (**obligatoire**) : entier, le chiffre/la classe à générer.

Version minimale (tous les défauts s'appliquent) :

```json
{ "dataframe_records": [ { "classe": 7 } ] }
```

## 5. Action `interpolate` — slider d'interpolation

### Requête

```json
{
  "dataframe_records": [
    { "action": "interpolate", "dataset": "mnist", "classe_a": 3, "classe_b": 8, "t": 0.5 }
  ]
}
```

- `classe_a`, `classe_b` (**obligatoires**) : les deux classes entre lesquelles interpoler.
- `t` (**obligatoire**) : position du slider, un nombre entre `0.0` (image de `classe_a`) et `1.0` (image de `classe_b`). `0.5` donne l'image à mi-chemin.
- `dataset` (optionnel, défaut `"mnist"`).

**Utilisation typique côté frontend :** un slider HTML `<input type="range" min="0" max="1" step="0.01">`, dont chaque changement de valeur déclenche un appel avec le `t` correspondant, pour ré-afficher l'image interpolée en direct.

## 6. Réponse (commune aux deux actions)

```json
{
  "predictions": [
    { "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..." }
  ]
}
```

`predictions` contient une entrée par ligne envoyée dans la requête, **dans le même ordre**. `image_base64` est une image **PNG encodée en base64**.

Plusieurs requêtes (générations et/ou interpolations mélangées) peuvent être envoyées en une seule fois :

```json
{
  "dataframe_records": [
    { "action": "generate", "classe": 7 },
    { "action": "interpolate", "classe_a": 3, "classe_b": 8, "t": 0.5 }
  ]
}
```

### Afficher l'image côté frontend

```html
<img src="data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAA..." />
```

Ou décoder en fichier/bytes côté backend (exemple Python) :

```python
import base64
image_bytes = base64.b64decode(response["predictions"][0]["image_base64"])
with open("digit.png", "wb") as f:
    f.write(image_bytes)
```

### Erreurs

Toutes les erreurs métier renvoient un message explicite dans le champ `message` (ou en fin de `stack_trace`) de la réponse d'erreur :
- `action='generate' nécessite le champ 'classe'`
- `action='interpolate' nécessite les champs 'classe_a', 'classe_b' et 't'`
- `classe doit être comprise entre 0 et 9 pour 'mnist', reçu 15`
- `t doit être compris entre 0 et 1, reçu 1.5`
- `dataset 'celeba' indisponible pour la génération. Datasets chargés : mnist` (la liste des datasets réellement chargés est toujours indiquée, pas besoin de la deviner)
- `action 'foo' inconnue. Valeurs possibles : 'generate', 'interpolate'`

Le backend web doit lire ce champ `message` pour l'afficher proprement à l'utilisateur plutôt que de renvoyer l'erreur brute.

## 7. Datasets disponibles

| Dataset | Valeur du champ `dataset` | `generate` (CVAE) | `interpolate` (VAE) |
|---|---|---|---|
| MNIST | `"mnist"` | Disponible | Disponible |
| Fashion-MNIST | `"fashion_mnist"` | Pas encore entraîné | Pas encore entraîné |
| CelebA | `"celeba"` | Pas encore entraîné | Pas encore entraîné |

Version actuelle du modèle enregistré : `image_generator` **v2**.

**Quand un nouveau dataset devient disponible :** la personne responsable ajoute son entrée dans `configs/deployment_registry.yaml` (chemins de ses configs CVAE et VAE, ses checkpoints, et ses centroïdes latents — voir [section 10](#10-comment-les-modèles-ont-été-enregistrés-pour-information--si-besoin-de-le-refaire)) et relance `python scripts/register_generation_model.py`, ce qui crée une nouvelle version du modèle. **Rien ne change côté application web** : même URL, même port, même format de requête — juste de nouvelles valeurs possibles pour `dataset`, et le numéro de version dans la commande `mlflow models serve` à mettre à jour.

## 8. Exemple complet

```bash
# Génération
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"action": "generate", "dataset": "mnist", "classe": 7}]}'

# Interpolation
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"action": "interpolate", "dataset": "mnist", "classe_a": 3, "classe_b": 8, "t": 0.5}]}'
```

```javascript
// exemple JavaScript (fetch) - génération
async function generate(classe) {
  const res = await fetch("http://127.0.0.1:5001/invocations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataframe_records: [{ action: "generate", classe }] }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.message);
  return data.predictions[0].image_base64;
}

// exemple JavaScript (fetch) - interpolation, appelé à chaque mouvement du slider
async function interpolate(classeA, classeB, t) {
  const res = await fetch("http://127.0.0.1:5001/invocations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataframe_records: [{ action: "interpolate", classe_a: classeA, classe_b: classeB, t }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.message);
  return data.predictions[0].image_base64;
}
```

## 9. Où faire tourner ce serveur

**Pour le développement et la soutenance, aucun VPS n'est nécessaire.** Le serveur MLflow tourne en local sur la machine qui fait la démonstration ; le backend web pointe simplement vers `http://localhost:5001` (ou l'IP de la machine sur le réseau local si frontend et backend sont sur des postes différents).

Un hébergement externe (VPS, ou plus simple : une plateforme gratuite comme Render/Railway/Hugging Face Spaces) ne devient utile que si l'application doit être accessible en continu sur internet, indépendamment de tout ordinateur du groupe. Ce n'est pas un prérequis du sujet.

## 10. Comment les modèles ont été enregistrés (pour information / si besoin de le refaire)

```bash
python scripts/compute_latent_centroids.py   # une fois par VAE entraîné, prérequis pour l'interpolation
python scripts/register_generation_model.py
```

`register_generation_model.py` :
1. Lit `configs/deployment_registry.yaml` pour savoir quels datasets ont un CVAE (et, si présent, un VAE + ses centroïdes latents) disponibles.
2. Charge chaque modèle disponible.
3. Enveloppe le tout dans un wrapper MLflow unique (`src/serving/generation_pyfunc.py`) qui route `predict({"action", "dataset", ...})` vers le bon modèle et la bonne action.
4. Enregistre ce wrapper comme une nouvelle version dans le Model Registry MLflow, sous le nom `image_generator`.

**Piège technique évité :** les checkpoints VAE et CVAE d'un même dataset partagent le même nom de fichier (`best_checkpoint.pth`). `build_artifacts()` les recopie d'abord dans un dossier de préparation avec des noms uniques avant de les transmettre à MLflow, pour éviter qu'un fichier n'en écrase un autre lors de l'enregistrement (bug rencontré et corrigé pendant le développement).

## 11. Testé et validé

Ce pipeline a été testé de bout en bout avec de vraies requêtes HTTP (`curl`) : génération (avec et sans champs optionnels), interpolation à plusieurs positions du slider (t=0, 0.25, 0.5, 0.75, 1 — vérifié visuellement que t=0 et t=1 correspondent bien aux deux classes demandées et que la transition est progressive), et tous les cas d'erreur listés en section 6.
