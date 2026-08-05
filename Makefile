install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python -m src.training.train --config configs/mnist_vae.yaml

ablation:
	python scripts/run_ablation.py --config configs/ablation_beta.yaml

test:
	python -m pytest

lint:
	python -m py_compile src/**/*.py

clean:
	rm -rf __pycache__ build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
