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

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

mlflow-backfill:
	python scripts/backfill_mlflow.py

register-cvae:
	python scripts/register_cvae_model.py

serve-cvae:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/cvae_generator/2" -p 5001 --env-manager local

test:
	python -m pytest

lint:
	python -m py_compile src/**/*.py

clean:
	rm -rf __pycache__ build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
