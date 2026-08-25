install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train-vae:
	python -m src.training.train --config configs/mnist_vae.yaml

train-cvae:
	python -m src.training.train --config configs/mnist_cvae.yaml

ablation:
	python scripts/run_ablation.py --config configs/ablation_beta.yaml

evaluate:
	python scripts/evaluate.py

test:
	python -m pytest

lint:
	python -m py_compile src/**/*.py

clean:
	rm -rf __pycache__ build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# ------------------------------------------------------------------ #
# Demonstration web (backend Flask + frontend React)
# ------------------------------------------------------------------ #

# Dependances de l'API seules (plus legeres que requirements.txt racine)
install-demo:
	python -m pip install -r backend/requirements.txt
	npm install --prefix frontend

# Echantillon d'images MNIST reelles utilise par la demo (~20 Ko)
fixtures:
	python scripts/build_demo_fixtures.py

# --- MLflow : le Model Registry alimente toute l'inference de l'API ---

# Empaquette les checkpoints disponibles et les enregistre. A relancer apres
# tout ajout de checkpoint : l'API ne voit que ce qui est au registre.
register:
	python scripts/register_models.py

register-list:
	python scripts/register_models.py --list

# Interface MLflow : experiences, parametres et Model Registry.
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# --- Developpement : deux process, rechargement a chaud du frontend ---

# Terminal 1
dev-api:
	python -m backend.app --port 8000 --reload

# Terminal 2 : Vite proxifie /api vers :8000
dev-web:
	npm run dev --prefix frontend

# --- Demonstration : un seul process, tout sur :8000 ---

demo: fixtures register
	npm run build --prefix frontend
	python -m backend.app --port 8000

# --- Demonstration dockerisee ---

# Tout en une commande : build de l'image puis demarrage du service.
docker-up:
	docker compose up -d --build
	@echo "Demo sur http://localhost:8000 — documentation d'API sur /docs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# A lancer apres chaque build : le Dockerfile tolere l'echec de
# l'enregistrement MLflow, ce qui peut produire une image qui demarre sans
# servir aucun modele.
docker-check:
	docker compose run --rm --no-deps demo python scripts/register_models.py --verify

# Enregistre les checkpoints CelebA deposes via le volume, puis redemarre pour
# que le catalogue les prenne en compte.
docker-register-celeba:
	docker compose exec demo python scripts/register_models.py --dataset celeba
	docker compose restart demo

.PHONY: install train-vae train-cvae ablation evaluate test lint clean \
        install-demo fixtures register register-list mlflow-ui dev-api dev-web demo \
        docker-up docker-down docker-logs docker-check docker-register-celeba
