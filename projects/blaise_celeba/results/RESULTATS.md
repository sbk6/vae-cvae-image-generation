# Resultats historiques. Etude d'ablation fine sur beta (beta-VAE), CelebA

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

La configuration actuelle d'ablation utilise maintenant 10 epochs et 8 000 images equilibrees par combinaison d'attributs. Ce fichier sera regenere quand `python -m evaluation.run_ablation --config configs/ablation_beta.yaml` sera relance.
