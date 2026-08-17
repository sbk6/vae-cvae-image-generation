# Image unique servant l'API Flask ET le frontend React buildé sur le meme
# port. Un seul container : moins de pieces mobiles le jour de la demo qu'un
# docker-compose a deux services.

# ---------------------------------------------------------------- frontend
FROM node:20-alpine AS frontend-build

WORKDIR /build

# Les dependances sont copiees seules d'abord : tant que package.json ne change
# pas, Docker reutilise le cache de npm ci au lieu de tout reinstaller.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------ runtime
# Python 3.11 pour coller a l'environnement de l'equipe modele (les .pyc du
# depot sont en cpython-311).
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependances Python en premier, pour la meme raison de cache que ci-dessus :
# c'est l'etape lente (~350 Mo de wheels torch CPU).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    # Le wheel torch CPU embarque ~145 Mo dont l'inference n'a aucun usage :
    # headers C++ (include/) et suite de tests (test/).
    #
    # Elagage volontairement conservateur : verifie par essai/erreur, tout le
    # reste est charge a l'import. Ne PAS retirer non plus :
    #   torch/bin      -> torch_shm_manager, cherche par torch/__init__.py
    #   torch/lib      -> les .so du runtime
    #   torchgen/      -> importe par torch.utils._python_dispatch
    && TORCH=/usr/local/lib/python3.11/site-packages/torch \
    && rm -rf "$TORCH/include" "$TORCH/test"

# Code de l'equipe modele, reutilise tel quel.
#   src/      -> modeles MNIST (Sylvain)
#   projects/ -> modeles Fashion-MNIST (David)
# Les deux arbres sont importes par les adaptateurs de backend/adapters/.
COPY src/ ./src/
COPY configs/ ./configs/
COPY projects/ ./projects/

# Code de l'API + fixtures d'images precalcules
COPY backend/ ./backend/

# Checkpoints entraines. Copies explicitement pour ne pas embarquer
# reports/figures/ ni les logs CSV.
COPY reports/experiments/ ./reports/experiments/

# Build React produit a l'etape precedente
COPY --from=frontend-build /build/dist ./frontend/dist

EXPOSE 8000

# 0.0.0.0 pour etre joignable depuis l'hote ; waitress gere le multi-thread.
CMD ["python", "-m", "backend.app", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
