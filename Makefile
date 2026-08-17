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

# --- Developpement : deux process, rechargement a chaud du frontend ---

# Terminal 1
dev-api:
	python -m backend.app --port 8000 --debug

# Terminal 2 : Vite proxifie /api vers :8000
dev-web:
	npm run dev --prefix frontend

# --- Demonstration : un seul process, tout sur :8000 ---

demo: fixtures
	npm run build --prefix frontend
	python -m backend.app --port 8000

# --- Demonstration dockerisee : un seul container ---

docker-build:
	docker build -t vae-cvae-demo .

docker-run:
	docker run --rm -p 8000:8000 vae-cvae-demo

# Variante montant les checkpoints Fashion-MNIST depuis l'hote : permet de
# remplacer les poids de David sans reconstruire l'image.
docker-run-mounted:
	docker run --rm -p 8000:8000 \
	  -v "$(CURDIR)/projects/david_fashion_mnist/checkpoints:/app/projects/david_fashion_mnist/checkpoints:ro" \
	  vae-cvae-demo

docker-demo: docker-build docker-run

.PHONY: install train-vae train-cvae ablation evaluate test lint clean \
        install-demo fixtures dev-api dev-web demo \
        docker-build docker-run docker-run-mounted docker-demo
