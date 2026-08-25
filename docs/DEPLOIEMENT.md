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
| CelebA | absent du dépôt, voir §6 |
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

| Usage | RAM | Disque |
|---|---|---|
| **Exécution seule** (image pré-construite) | 1,5 Go suffit, 2 Go confortable | 4 Go |
| **Build sur le VPS** | **4 Go minimum** | 10 Go |

Le build est bien plus gourmand que l'exécution : installation de torch
(~350 Mo de wheels décompressés sur place) et build npm du frontend. Sur un
VPS à 2 Go, le build échoue typiquement par OOM pendant `pip install torch`.

**D'où la recommandation : construire l'image ailleurs (§5) et ne faire que la
tirer sur le VPS.** Un VPS à 2 Go suffit alors largement.

---

## 3. Préparer le VPS (Ubuntu 22.04 / 24.04)

```bash
ssh utilisateur@IP_DU_VPS
```

Installer Docker depuis le dépôt officiel — la version des dépôts Ubuntu est
souvent trop ancienne :

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Pouvoir lancer Docker sans `sudo` :

```bash
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world
```

---

## 4. Voie A — build directement sur le VPS

À réserver aux VPS d'au moins 4 Go de RAM.

### 4.1 Cloner

```bash
git clone https://github.com/sbk6/vae-cvae-image-generation.git
cd vae-cvae-image-generation
```

La branche par défaut est `main`, aucune option nécessaire.

Vérifier que les poids attendus sont bien là :

```bash
find reports/experiments projects -name "*.pth" -o -name "*.pt" | grep -v packaged_models
```

Sept fichiers doivent apparaître. S'il n'y en a aucun, le dépôt a été cloné en
`--filter` ou en shallow partiel : recloner sans option.

### 4.2 Construire

```bash
docker build -t vae-demo .
```

Compter 5 à 15 minutes selon la bande passante. L'étape lente est
`pip install torch`.

### 4.3 Vérifier que le registre MLflow a bien été peuplé

**Étape à ne pas sauter.** Le Dockerfile tolère l'échec de l'enregistrement
pour que le build aboutisse quand même — un registre vide produirait donc une
image qui démarre mais ne sert aucun modèle.

```bash
docker run --rm vae-demo python -c "
from backend.mlflow_registry import RegistryGateway
noms = RegistryGateway().available_names()
print(len(noms), 'modeles enregistres')
[print(' ', n) for n in noms]
"
```

Sortie attendue : **7 modèles**. Si la sortie affiche `0`, voir §8.

### 4.4 Lancer

```bash
docker run -d \
  --name vae-demo \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  vae-demo
```

`127.0.0.1:8000` et non `-p 8000:8000` : le container n'est joignable que
localement, le reverse proxy s'occupe de l'exposition publique. Publier
directement le port contournerait le pare-feu UFW, qui ne filtre pas les
règles créées par Docker.

### 4.5 Vérifier

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

---

## 5. Voie B — CI/CD avec GitHub Actions (recommandée)

L'image est construite par GitHub Actions, publiée sur le registre de conteneurs
GitHub (GHCR), et le VPS ne fait que la tirer. Avantages : un VPS à 2 Go
suffit, le build ne monopolise plus la machine de production, et chaque push
sur `main` produit une image traçable.

### 5.1 Le workflow

Créer `.github/workflows/deploy.yml` :

```yaml
name: Build et publier l'image

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Se connecter a GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Construire et publier
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Verifier que le registre MLflow est peuple
        run: |
          docker run --rm ghcr.io/${{ github.repository }}:${{ github.sha }} \
            python -c "
          from backend.mlflow_registry import RegistryGateway
          n = len(RegistryGateway().available_names())
          print(n, 'modeles enregistres')
          assert n >= 7, f'Registre incomplet : {n} modeles'
          "
```

Le tag `:${{ github.sha }}` permet de revenir en arrière sur une version
précise ; `:latest` sert au déploiement courant.

L'étape de vérification fait **échouer la CI** si le registre est vide — c'est
précisément le mode de panne silencieux que le `|| echo` du Dockerfile
introduit.

### 5.2 Rendre le paquet accessible

Par défaut, un paquet GHCR est privé. Deux possibilités :

- **Le rendre public** : page du dépôt → onglet *Packages* → le paquet →
  *Package settings* → *Change visibility* → *Public*. Le VPS tire alors sans
  authentification.
- **Le garder privé** : créer un *Personal Access Token* (classic) avec la
  portée `read:packages`, puis sur le VPS :

```bash
echo "LE_TOKEN" | docker login ghcr.io -u VOTRE_LOGIN_GITHUB --password-stdin
```

### 5.3 Déployer sur le VPS

```bash
docker pull ghcr.io/sbk6/vae-cvae-image-generation:latest

docker rm -f vae-demo 2>/dev/null || true

docker run -d \
  --name vae-demo \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  ghcr.io/sbk6/vae-cvae-image-generation:latest
```

### 5.4 Script de mise à jour

Créer `~/deploy.sh` sur le VPS :

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/sbk6/vae-cvae-image-generation:latest"

echo "Recuperation de l'image..."
docker pull "$IMAGE"

echo "Redemarrage du container..."
docker rm -f vae-demo 2>/dev/null || true
docker run -d --name vae-demo --restart unless-stopped \
  -p 127.0.0.1:8000:8000 "$IMAGE"

echo "Attente de l'API..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/api/health > /dev/null; then
    echo "En ligne. Modeles servis :"
    curl -s http://127.0.0.1:8000/api/health \
      | python3 -c "import json,sys; print(' ', len(json.load(sys.stdin)['registered_models']))"
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

### 5.5 Déploiement automatique (facultatif)

Pour que le VPS se mette à jour à chaque push, ajouter un job au workflow. Il
faut d'abord créer les secrets `VPS_HOST`, `VPS_USER` et `VPS_SSH_KEY` dans
*Settings → Secrets and variables → Actions*.

```yaml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deployer par SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: ./deploy.sh
```

Utiliser une clé SSH **dédiée à ce déploiement**, pas une clé personnelle.

---

## 6. Ajouter les modèles CelebA

Les poids CelebA ne sont pas dans le dépôt — 100 Mo par fichier, au-dessus de
la limite GitHub. Ils doivent être copiés sur le VPS et enregistrés à la main.

Le registre MLflow étant construit au build, il faut **le persister sur un
volume** pour que l'enregistrement survive à un redémarrage.

### 6.1 Copier les poids

Depuis votre machine :

```bash
scp best_checkpoint.pth utilisateur@IP_DU_VPS:~/celeba/
```

### 6.2 Lancer avec les volumes

```bash
docker volume create vae-mlflow

docker rm -f vae-demo 2>/dev/null || true

docker run -d \
  --name vae-demo \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v ~/celeba:/app/projects/blaise_celeba/results/experiments/cvae_improved:ro \
  -v vae-mlflow:/app/mlartifacts \
  ghcr.io/sbk6/vae-cvae-image-generation:latest
```

### 6.3 Enregistrer

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

## 7. Exposer publiquement — Nginx et HTTPS

### 7.1 Nginx

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
        # secondes : le timeout par defaut de 60 s convient, mais on le rend
        # explicite pour eviter les surprises sur un VPS lent.
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/vae-demo /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 7.2 Pare-feu

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

Le port 8000 ne doit **pas** être ouvert : le container n'écoute que sur
`127.0.0.1`.

### 7.3 Certificat HTTPS

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

## 8. Vérifications et pannes courantes

### L'API répond mais aucun modèle n'est servi

```bash
docker exec vae-demo python -c "
from backend.mlflow_registry import RegistryGateway, is_registry_available
print('registre present :', is_registry_available())
print('modeles :', RegistryGateway().available_names())
"
```

Si la liste est vide, l'enregistrement a échoué au build. Le refaire dans le
container tournant :

```bash
docker exec vae-demo python scripts/register_models.py
docker restart vae-demo
```

Attention : sans volume sur `/app/mlartifacts`, cet enregistrement est perdu au
prochain `docker run`. Reconstruire l'image proprement est préférable.

### Le build échoue pendant `pip install torch`

Presque toujours un manque de mémoire. Vérifier :

```bash
free -h
```

Ajouter du swap temporairement :

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Ou passer par la voie B (§5), qui supprime le problème.

### La première requête met 10 secondes

Comportement normal : c'est l'initialisation de MLflow au premier chargement de
modèle. L'application lance un préchauffage en arrière-plan au démarrage, donc
attendre une trentaine de secondes après le lancement avant de tester.

### Journaux

```bash
docker logs -f vae-demo
docker stats vae-demo        # memoire et CPU en direct
```

### Redémarrage complet

```bash
docker restart vae-demo
```

---

## 9. Limites connues

**CelebA n'est pas dans l'image.** Ses poids pèsent 100 Mo par fichier, exclus
du dépôt. Procédure manuelle au §6.

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

## 10. Ce qui n'a pas pu être vérifié

Cette procédure décrit le `Dockerfile` du dépôt et des mesures réelles prises
sur l'application. En revanche, **le build complet depuis un clone frais n'a
pas pu être rejoué** : le daemon Docker de la machine de rédaction est tombé
pendant la vérification.

Deux points à confirmer au premier déploiement :

1. l'étape `RUN python scripts/register_models.py` du Dockerfile aboutit bien
   dans le contexte du build (§4.3 la vérifie explicitement) ;
2. la taille finale de l'image — estimée à ~1,4 Go, mesurée à 1,33 Go avant
   l'ajout du registre MLflow.

```bash
docker images vae-demo --format "{{.Size}}"
```
