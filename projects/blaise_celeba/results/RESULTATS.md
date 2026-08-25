# Resultats CelebA disponibles

## Modeles principaux Colab

Les poids Colab du protocole ameliore sont disponibles dans les chemins
attendus par la demo :

```text
projects/blaise_celeba/results/experiments/vae_improved/best_checkpoint.pth
projects/blaise_celeba/results/experiments/cvae_improved/best_checkpoint.pth
```

Protocole : 32 000 images train, 3 000 validation, 3 000 test, sampling
`balanced_conditions`, latent_dim 128, hidden_channels 64, beta final 0.5 avec
KL annealing. Les runs etaient configures pour 100 epochs max, mais l'early
stopping retarde apres l'epoch 20 les a arretes a l'epoch 30.

| Modele | best epoch | last epoch | loss val best | reconstruction val best | KL val best | beta best |
|---|---:|---:|---:|---:|---:|---:|
| VAE | 4 | 30 | 752.04 | 661.03 | 455.07 | 0.20 |
| CVAE | 3 | 30 | 600.99 | 512.78 | 588.07 | 0.15 |

Resultats test issus de `results/experiments/comparison.json` :

| Modele | reconstruction test | KL test | n_test |
|---|---:|---:|---:|
| VAE | 650.45 | 376.39 | 3 000 |
| CVAE | 664.25 | 218919.49 | 3 000 |

Note : les checkpoints principaux ont ete normalises pour contenir
`configuration.model.type` (`vae` ou `cvae`), attendu par l'adaptateur de la
demo.

---

## Resultats historiques. Etude d'ablation fine sur beta (beta-VAE), CelebA

Protocole deja execute avant le dernier changement de configuration : VAE identique pour toutes les valeurs de beta (latent_dim=64, hidden_channels=32, epochs=40, seed=42, n_train=4000).

| beta | meilleure epoch | reconstruction (val) | KL (val) | loss totale (val) |
|---|---|---|---|---|
| 0.05 | 32 | 503.55 | 390.73 | 523.08 |
| 0.1 | 28 | 520.54 | 302.39 | 550.78 |
| 0.25 | 28 | 543.64 | 212.89 | 596.86 |
| 0.5 | 32 | 558.25 | 169.10 | 642.80 |
| 0.75 | 24 | 576.23 | 143.14 | 683.58 |
| 1.0 | 24 | 584.35 | 129.38 | 713.74 |
| 1.5 | 24 | 608.45 | 108.91 | 771.82 |
| 2.0 | 32 | 618.87 | 101.30 | 821.48 |

Lecture : quand beta augmente, la KL est davantage penalisee, donc elle baisse (latent plus proche de la loi normale, mieux regularise), mais la reconstruction se degrade (image reconstruite moins fidele). Un beta trop petit fait l'inverse : bonne reconstruction, mais latent peu structure, avec un risque de mauvaise generation quand on echantillonne un z aleatoire.

La configuration actuelle d'ablation utilise maintenant 100 epochs maximum et 8 000 images equilibrees par combinaison d'attributs, avec arret anticipe. Ce fichier sera regenere quand `python -m evaluation.run_ablation --config configs/ablation_beta.yaml` sera relance.
