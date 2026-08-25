# Déploiement sur VPS — procédure complète

Démo VAE / CVAE : API FastAPI + interface React, servies sur un seul port, avec
inférence via le MLflow Model Registry.

Dépôt : `https://github.com/sbk6/vae-cvae-image-generation` — branche `main`.

---

## 1. Ce qui est déployé

Un seul container. Le frontend React est buildé puis servi par l'API, donc pas
de second service à orchestrer.

| | |
|---|---|
| Port exposé | `8000` |
| Image | `python:3.11-slim` + torch CPU, **~1,4 Go** |
| Modèles servis | 7 depuis le dépôt (5 MNIST + 2 Fashion-MNIST) |
| CelebA | absent du dépôt, voir §7 |
| Base de données | aucune (SQLite MLflow embarqué) |
| GPU | inutile, tout tourne sur CPU |

**Le Model Registry MLflow est construit pendant le build de l'image**, pas au
démarrage. Un container qui démarre sert donc immédiatement ce qui était
présent au moment du build.

---

## 2. Dimensionner le VPS

Mesures réelles du processus Python, pas des estimations :

| Étape | Mémoire résidente |
|---|---|
| Processus nu | 15 Mo |
| + torch importé | 188 Mo |
| + mlflow importé | 284 Mo |
| Application démarrée | **471 Mo** |
| Les 8 modèles chargés en cache | **565 Mo** |

Le premier chargement d'un modèle prend **~9 secondes** (initialisation
MLflow) ; les suivants sont sous la seconde puis mis en cache.

| Besoin | RAM | Disque |
|---|---|---|
| Faire tourner l'application | 1,5 Go suffit | 4 Go |
| **Construire l'image** | **4 Go** | 10 Go |

> **Le build est bien plus gourmand que l'exécution.** Installation de torch
> (~350 Mo de wheels décompressés sur place) et build npm du frontend. Sur un
> VPS à 2 Go, le build échoue typiquement par manque de mémoire pendant
> `pip install torch`.
>
> Si le VPS a moins de 4 Go, **ajouter du swap avant de construire** (§3.3).
> Une fois l'image construite, le swap peut être retiré : l'exécution n'en a
> pas besoin.

---

## 3. Préparer le VPS (Ubuntu 22.04 / 24.04)

### 3.1 Se connecter

```bash
ssh utilisateur@IP_DU_VPS
```

### 3.2 Installer Docker

Depuis le dépôt officiel — la version des dépôts Ubuntu est souvent trop
ancienne :

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

Pouvoir lancer Docker sans `sudo` :

```bash
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
```

### 3.3 Ajouter du swap si le VPS a moins de 4 Go

Vérifier d'abord :

```bash
free -h
```

Si la ligne `Mem` affiche moins de 4 Go :

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

Pour le rendre permanent (facultatif — le swap n'est utile qu'au build) :

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 4. Récupérer et construire

### 4.1 Cloner

```bash
git clone https://github.com/sbk6/vae-cvae-image-generation.git
cd vae-cvae-image-generation
```

La branche par défaut est `main`, aucune option nécessaire.

Vérifier que les poids attendus sont bien présents :

```bash
find reports/experiments projects -name "*.pth" -o -name "*.pt" | grep -v packaged_models
```

Sept fichiers doivent apparaître. S'il n'y en a aucun, le dépôt a été cloné en
shallow partiel : recloner sans option.

### 4.2 Construire

```bash
docker build -t vae-demo .
```

Compter 5 à 15 minutes selon la bande passante. L'étape lente est
`pip install torch`.

### 4.3 Vérifier que le registre MLflow a bien été peuplé

**Étape à ne pas sauter.** Le Dockerfile tolère volontairement l'échec de
l'enregistrement pour que le build aboutisse quand même. Sans cette
vérification, on obtiendrait une image qui démarre normalement mais ne sert
aucun modèle — une panne silencieuse.

```bash
docker run --rm vae-demo python -c "
from backend.mlflow_registry import RegistryGateway
noms = RegistryGateway().available_names()
print(len(noms), 'modeles enregistres')
[print(' ', n) for n in noms]
"
```

Sortie attendue : **7 modèles**. Si la sortie affiche `0`, voir §8.

---

## 5. Lancer

```bash
docker run -d \
  --name vae-demo \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  vae-demo
```

Deux options importantes :

- `--restart unless-stopped` : le container repart tout seul après un
  redémarrage du VPS.
- `-p 127.0.0.1:8000:8000` et non `-p 8000:8000` : le container n'est joignable
  que localement, le reverse proxy s'occupe de l'exposition publique. Publier
  directement le port contournerait le pare-feu UFW, qui ne filtre pas les
  règles créées par Docker.

### Vérifier

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

Réponse attendue :

```json
{
  "status": "ok",
  "inference": "mlflow-registry",
  "registry_available": true,
  "registered_models": ["mnist-vae_main", "..."],
  "datasets": ["mnist", "fashion_mnist"]
}
```

Laisser une trentaine de secondes après le lancement avant de tester :
l'application précharge un modèle en arrière-plan au démarrage.

---

## 6. Mettre à jour après un nouveau commit

```bash
cd ~/vae-cvae-image-generation
git pull
docker build -t vae-demo .
docker rm -f vae-demo
docker run -d --name vae-demo --restart unless-stopped \
  -p 127.0.0.1:8000:8000 vae-demo
```

Pour éviter de retaper la séquence, créer `~/deploy.sh` :

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/vae-cvae-image-generation
git pull

docker build -t vae-demo .

# Verification bloquante : une image sans registre demarre sans rien servir.
docker run --rm vae-demo python -c "
from backend.mlflow_registry import RegistryGateway
n = len(RegistryGateway().available_names())
print(n, 'modeles enregistres')
assert n >= 7, 'Registre incomplet, deploiement interrompu'
"

docker rm -f vae-demo 2>/dev/null || true
docker run -d --name vae-demo --restart unless-stopped \
  -p 127.0.0.1:8000:8000 vae-demo

echo "Attente de l'API..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/api/health > /dev/null; then
    echo "En ligne."
    exit 0
  fi
  sleep 2
done

echo "L'API n'a pas repondu en 60 s. Journaux :"
docker logs --tail 40 vae-demo
exit 1
```

```bash
chmod +x ~/deploy.sh
~/deploy.sh
```

Nettoyer les anciennes images de temps en temps, elles pèsent 1,4 Go pièce :

```bash
docker image prune -f
```

---

## 7. Ajouter les modèles CelebA

Les poids CelebA ne sont pas dans le dépôt — 100 Mo par fichier, au-dessus de
la limite GitHub. Ils doivent être copiés sur le VPS et enregistrés à la main.

Le registre MLflow étant construit au build, il faut **le persister sur un
volume** pour que l'enregistrement survive à un redémarrage.

### 7.1 Copier les poids

Depuis votre machine :

```bash
scp best_checkpoint.pth utilisateur@IP_DU_VPS:~/celeba/
```

### 7.2 Relancer avec les volumes

```bash
docker volume create vae-mlflow

docker rm -f vae-demo 2>/dev/null || true

docker run -d \
  --name vae-demo \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v ~/celeba:/app/projects/blaise_celeba/results/experiments/cvae_improved:ro \
  -v vae-mlflow:/app/mlartifacts \
  vae-demo
```

### 7.3 Enregistrer

```bash
docker exec vae-demo python scripts/register_models.py --dataset celeba
docker restart vae-demo
```

Le redémarrage est nécessaire : le catalogue est construit au démarrage de
l'application.

Vérifier :

```bash
curl -s http://127.0.0.1:8000/api/datasets | python3 -m json.tool
```

`celeba` doit apparaître avec `model_count` supérieur à zéro.

---

## 8. Exposer publiquement — Nginx et HTTPS

### 8.1 Nginx

```bash
sudo apt-get install -y nginx
sudo nano /etc/nginx/sites-available/vae-demo
```

```nginx
server {
    listen 80;
    server_name demo.votre-domaine.fr;

    # Les images sont des data-URI base64 : une reponse de generation
    # peut depasser le tampon par defaut.
    client_max_body_size 10M;
    proxy_buffer_size 16k;
    proxy_buffers 8 16k;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Le premier chargement d'un modele MLflow prend une dizaine de
        # secondes : on rend le delai explicite pour eviter les surprises
        # sur un VPS lent.
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/vae-demo /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 8.2 Pare-feu

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

Le port 8000 ne doit **pas** être ouvert : le container n'écoute que sur
`127.0.0.1`.

### 8.3 Certificat HTTPS

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d demo.votre-domaine.fr
```

Certbot modifie la configuration Nginx et installe le renouvellement
automatique. Vérifier :

```bash
sudo certbot renew --dry-run
```

---

## 9. Pannes courantes

### L'API répond mais aucun modèle n'est servi

```bash
docker exec vae-demo python -c "
from backend.mlflow_registry import RegistryGateway, is_registry_available
print('registre present :', is_registry_available())
print('modeles :', RegistryGateway().available_names())
"
```

Si la liste est vide, l'enregistrement a échoué au build. Le refaire dans le
container en marche :

```bash
docker exec vae-demo python scripts/register_models.py
docker restart vae-demo
```

Attention : sans volume sur `/app/mlartifacts`, cet enregistrement est perdu au
prochain `docker run`. Reconstruire l'image proprement est préférable.

### Le build échoue pendant `pip install torch`

Presque toujours un manque de mémoire :

```bash
free -h
```

Ajouter du swap comme au §3.3, puis relancer le build.

### La première requête met 10 secondes

Comportement normal : c'est l'initialisation de MLflow au premier chargement de
modèle. Attendre une trentaine de secondes après le lancement avant de tester.

### Journaux et ressources

```bash
docker logs -f vae-demo
docker stats vae-demo        # memoire et CPU en direct
```

### Redémarrage complet

```bash
docker restart vae-demo
```

---

## 10. Limites connues

**CelebA n'est pas dans l'image.** Ses poids pèsent 100 Mo par fichier, exclus
du dépôt. Procédure manuelle au §7.

**L'ablation β est inactive sur Fashion-MNIST et CelebA.** Seul le run β = 1 a
été livré pour ces datasets ; l'écran l'explique et affiche les tableaux de
résultats à la place.

**Le registre MLflow est figé au build.** Ajouter un checkpoint impose soit de
reconstruire l'image, soit de réenregistrer dans le container avec un volume
persistant.

**Pas de `docker-compose.yml`.** Un seul service, donc pas indispensable —
mais l'énoncé du cours attend `docker compose up --build` comme commande
unique. À ajouter avant le rendu.

---

## 11. Ce qui n'a pas pu être vérifié

Cette procédure décrit le `Dockerfile` du dépôt et s'appuie sur des mesures
réelles prises sur l'application. En revanche, **le build complet depuis un
clone frais n'a pas pu être rejoué** : le daemon Docker de la machine de
rédaction a cessé de répondre pendant la vérification.

Deux points à confirmer au premier déploiement :

1. l'étape `RUN python scripts/register_models.py` du Dockerfile aboutit bien
   dans le contexte du build — le §4.3 la vérifie explicitement ;
2. la taille finale de l'image, estimée à ~1,4 Go et mesurée à 1,33 Go avant
   l'ajout de l'étape d'enregistrement MLflow.

```bash
docker images vae-demo --format "{{.Size}}"
```
