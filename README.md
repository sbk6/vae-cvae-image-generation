# VAE / CVAE Image Generation — Projet

Résumé rapide (FR) : implémentation PyTorch d'un VAE et d'un CVAE, configurables par YAML, testés sur MNIST, avec scripts d'entraînement et de génération rapide (smoke-test).

## Architecture du dépôt
- `configs/` : paramètres YAML pour runs (dataset, training, modèle).
- `src/` : code source principal.
  - `src/data` : construction des dataloaders
  - `src/models` : `vae.py`, `cvae.py`, blocs réutilisables
  - `src/losses` : `elbo.py`
  - `src/training` : `trainer.py`, `train.py`
  - `src/utils` : `config.py`, `seed.py`
- `scripts/` : utilitaires (génération de grilles, inspection dataloader).
- `tests/` : tests unitaires pytest.
- `reports/figures` : images sauvegardées (grilles réelles et générées).
- `docs/` : documentation explicative (fichier `explanations.md`).

## Commandes utiles
Installer :

```bash
python -m pip install -r requirements.txt
```

Tests :

```bash
python -m pytest -q
```

Smoke-test d'entraînement VAE :

```bash
python -m src.training.train --config configs/mnist_vae.yaml --smoke-test
```

Générer grille CVAE :

```bash
python scripts/generate_cvae_grid.py --config configs/mnist_cvae.yaml --output reports/figures/cvae_grid.png --samples-per-class 8
```

## Interprétation des résultats
Voir `docs/explanations.md` pour une explication détaillée et pédagogique.

---

Si tu veux que je pousse la branche sur GitHub, fournis le remote (ex. `git@github.com:username/repo.git`) ou donne-moi l'autorisation d'utiliser un remote existant.
