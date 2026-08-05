# Plan d'exécution

Je vais suivre exactement les 9 étapes demandées pour construire le socle du projet VAE/CVAE sur MNIST, en respectant les contraintes suivantes : PyTorch uniquement, code agnostique au jeu de données, conditionnement générique, reproductibilité, documentation et structure claire.

## Étapes

1. Socle du projet
   - Créer `requirements.txt`
   - Créer `Makefile` avec les cibles `install`, `train`, `ablation`, `test`, `lint`, `clean`
   - Implémenter `src/utils/seed.py`
   - Implémenter `src/utils/config.py` pour charger YAML et surcharger par arguments CLI
   - Vérifier avec `make install` puis `make test`

2. Chargement des données
   - Implémenter `src/data/datasets.py`
   - Exposer `build_dataloaders(config)` renvoyant train/val/test et métadonnées dataset
   - Prévoir Fashion-MNIST et CelebA avec commentaires d'extension
   - Vérifier avec un script court qui affiche un batch et sauvegarde une planche de 16 images réelles

3. Le VAE
   - Créer `src/models/layers.py` pour blocs convolutifs partagés
   - Créer `src/models/vae.py` avec encodeur, reparamétrisation, décodeur
   - Créer `src/losses/elbo.py` avec reconstruction + KL et coefficient beta
   - Documenter en français l'origine des formules et la reparamétrisation
   - Vérifier avec `tests/test_shapes.py` et `tests/test_losses.py`

4. Boucle d'entraînement
   - Créer `src/training/trainer.py`
   - Boucle epochs, validation, checkpointing, CSV log, GPU, barres de progression
   - Créer `src/training/train.py` CLI avec `--smoke-test`
   - Vérifier avec `python -m src.training.train --config configs/mnist_vae.yaml --smoke-test`

5. Le CVAE
   - Créer `src/models/cvae.py`
   - Injecter condition dans encodeur et concaténer avant décodeur
   - Supporter `one_hot` et `multi_label`
   - Exposer `sample(condition, n)` et `reconstruct(x, condition)`
   - Vérifier avec entraînement court MNIST et planche de 10x8

6. Étude d'ablation sur beta
   - Créer `scripts/run_ablation.py`
   - Utiliser `configs/ablation_beta.yaml` avec 5 betas et 2 seeds
   - Agréger Markdown dans `docs/RESULTATS.md`
   - Ajouter courbe reconstruction vs KL
   - Vérifier que le tableau est généré et traçable

7. Analyse de l'espace latent
   - Créer `src/visualization/latent.py` pour t-SNE et UMAP
   - Créer `src/visualization/interpolation.py`
   - Créer `src/visualization/grids.py`
   - Sauvegarder figures dans `reports/figures/`
   - Vérifier la génération des figures

8. Évaluation chiffrée
   - Créer `src/evaluation/metrics.py`
   - Créer `src/evaluation/controllability.py`
   - Mesurer commentaires et comparatif VAE/CVAE
   - Ajouter tableau comparatif dans `docs/RESULTATS.md`

9. Documentation
   - Rédiger `docs/EXPLICATIONS.md` en français
   - Mettre à jour `README.md`

## Contraintes générales

- Pas de valeurs codées en dur pour les jeux de données.
- Pas de notebooks pour l'entraînement : seulement pour affichage des résultats.
- Dépendances minimales : torch, torchvision, numpy, matplotlib, pyyaml, scikit-learn, umap-learn, pytest.
- Comments et docs en français, variables et fichiers en anglais.
