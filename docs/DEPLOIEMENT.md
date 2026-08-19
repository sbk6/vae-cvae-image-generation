# Déploiement du modèle CVAE avec MLflow — guide pour l'équipe web

Ce document explique comment démarrer le service de génération d'images et comment l'appeler depuis un backend (Flask, FastAPI, Node, etc.). Il est destiné à la personne qui développe l'application web de démonstration ; elle n'a besoin de connaître ni PyTorch, ni MLflow, ni l'architecture des modèles.

## 1. Un seul endpoint pour tous les datasets

L'équipe travaille sur 3 datasets (MNIST, Fashion-MNIST, CelebA), chacun avec son propre CVAE entraîné séparément. **Il n'y a cependant qu'un seul serveur, un seul port, une seule URL à utiliser** : le champ `dataset` dans la requête indique simplement quel modèle utiliser. Pas besoin de gérer 3 ports différents côté application web, même si les 3 modèles ne sont pas encore tous entraînés au même moment.

```
[Frontend]  ->  [Backend web]  ->  HTTP POST /invocations  ->  [Serveur MLflow - un seul port]
                                                                     |
                                                         route en interne vers le bon
                                                         modele selon "dataset"
                                                                     |
                                                              image generee
```

À ce jour, seul **MNIST** est disponible (voir [section 5](#5-datasets-disponibles)). Si vous appelez le endpoint sans préciser `dataset`, il utilise MNIST par défaut.

## 2. Démarrer le serveur

Depuis la racine de ce dépôt (nécessite que `mlflow.db` et le modèle enregistré soient présents — voir `scripts/register_cvae_model.py` si besoin de le régénérer) :

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/cvae_generator/2" -p 5001 --env-manager local
```

- `-m "models:/cvae_generator/2"` : `2` est le numéro de version actuel (voir [section 5](#5-datasets-disponibles) pour la dernière version disponible et ce qu'elle contient).
- `-p 5001` : port d'écoute (à adapter si besoin).
- `--env-manager local` : utilise l'environnement Python déjà installé (plus rapide pour le développement local).

Le serveur met quelques dizaines de secondes à démarrer. Il est prêt quand :

```bash
curl http://127.0.0.1:5001/ping
```

répond `200 OK`. Les logs de démarrage indiquent aussi la liste des datasets effectivement chargés, par exemple :
```
[cvae_pyfunc] datasets disponibles : ['mnist']
```

## 3. Endpoints exposés

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/invocations` | POST | Génère une image. C'est l'endpoint principal. |
| `/ping` | GET | Vérifie que le serveur est démarré et prêt. |
| `/version` | GET | Version de MLflow utilisée. |

## 4. Contrat de l'endpoint `/invocations`

### Requête

```
POST /invocations
Content-Type: application/json
```

Corps de la requête — une entrée par image demandée :

```json
{
  "dataframe_records": [
    { "dataset": "mnist", "classe": 7 }
  ]
}
```

- `classe` (obligatoire) : entier, le chiffre/la classe à générer.
- `dataset` (optionnel, défaut `"mnist"`) : quel modèle utiliser parmi ceux disponibles (section 5).

Pour plusieurs images en une seule requête :

```json
{
  "dataframe_records": [
    { "dataset": "mnist", "classe": 7 },
    { "dataset": "mnist", "classe": 3 }
  ]
}
```

Sans préciser `dataset` (utilise MNIST par défaut) :

```json
{ "dataframe_records": [ { "classe": 7 } ] }
```

### Réponse

```json
{
  "predictions": [
    { "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..." }
  ]
}
```

`predictions` contient une entrée par image demandée, **dans le même ordre** que la requête. `image_base64` est une image **PNG encodée en base64**.

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

- **Classe hors limites** (ex. classe 15 sur un dataset à 10 classes) : erreur avec le message `classe doit être comprise entre 0 et 9 pour 'mnist', reçu 15`.
- **Dataset non disponible** (pas encore entraîné/déployé) : erreur avec le message `dataset 'celeba' indisponible. Datasets chargés actuellement : mnist` — la liste des datasets réellement chargés est toujours indiquée dans le message, pas besoin de la deviner.
- Dans les deux cas, MLflow renvoie un code HTTP d'erreur avec un champ `message` (et `stack_trace` en développement) contenant le message ci-dessus — le backend web doit lire `message` pour l'afficher proprement à l'utilisateur plutôt que de renvoyer l'erreur brute.

## 5. Datasets disponibles

| Dataset | Valeur du champ `dataset` | Classes | Statut |
|---|---|---|---|
| MNIST | `"mnist"` | 0 à 9 (chiffres) | Disponible (version 2 du modèle enregistré) |
| Fashion-MNIST | `"fashion_mnist"` | 0 à 9 (catégories de vêtements) | Pas encore entraîné |
| CelebA | `"celeba"` | attributs multiples | Pas encore entraîné |

**Quand un nouveau dataset devient disponible :** la personne responsable ajoute son entrée dans `configs/deployment_registry.yaml` (chemin de sa config + de son checkpoint) et relance `python scripts/register_cvae_model.py`, ce qui crée une nouvelle version du modèle (ex. version 3) incluant ce dataset en plus. **Rien ne change côté application web** : même URL, même port, même format de requête — juste une nouvelle valeur possible pour `dataset`, et le numéro de version dans la commande `mlflow models serve` à mettre à jour. Cette page sera mise à jour avec le nouveau numéro de version dès qu'un dataset est ajouté.

## 6. Exemple complet

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"dataset": "mnist", "classe": 7}]}'
```

```javascript
// exemple JavaScript (fetch)
const res = await fetch("http://127.0.0.1:5001/invocations", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ dataframe_records: [{ dataset: "mnist", classe: 7 }] }),
});
const data = await res.json();
if (!res.ok) {
  console.error("Erreur:", data.message);
} else {
  const imgBase64 = data.predictions[0].image_base64;
  document.getElementById("result").src = `data:image/png;base64,${imgBase64}`;
}
```

## 7. Où faire tourner ce serveur

**Pour le développement et la soutenance, aucun VPS n'est nécessaire.** Le serveur MLflow tourne en local sur la machine qui fait la démonstration ; le backend web pointe simplement vers `http://localhost:5001` (ou l'IP de la machine sur le réseau local si frontend et backend sont sur des postes différents).

Un hébergement externe (VPS, ou plus simple : une plateforme gratuite comme Render/Railway/Hugging Face Spaces) ne devient utile que si l'application doit être accessible en continu sur internet, indépendamment de tout ordinateur du groupe. Ce n'est pas un prérequis du sujet.

## 8. Comment le modèle a été enregistré (pour information / si besoin de le refaire)

```bash
python scripts/register_cvae_model.py
```

Ce script :
1. Lit `configs/deployment_registry.yaml` pour savoir quels datasets ont un checkpoint disponible.
2. Charge chaque CVAE entraîné disponible.
3. Enveloppe le tout dans un wrapper MLflow unique (`src/serving/cvae_pyfunc.py`) qui route `predict({"dataset", "classe"})` vers le bon modèle.
4. Enregistre ce wrapper comme une nouvelle version dans le Model Registry MLflow, sous le nom `cvae_generator`.

## 9. Testé et validé

Ce pipeline a été testé de bout en bout avec de vraies requêtes HTTP (`curl`) : génération avec et sans le champ `dataset` (défaut correct sur MNIST), et vérification que demander un dataset non disponible renvoie bien un message d'erreur clair plutôt qu'un plantage silencieux. Les images générées pour plusieurs classes ont été décodées et vérifiées visuellement.
