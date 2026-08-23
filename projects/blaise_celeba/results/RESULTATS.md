# Resultats baseline. Etude d'ablation sur beta (beta-VAE), CelebA

Protocole baseline deja execute : VAE identique pour toutes les valeurs de beta (latent_dim=64, hidden_channels=32, epochs=10, seed=42, n_train=4000).

| beta | reconstruction (val) | KL (val) | loss totale (val) |
|---|---|---|---|
| 0.1 | 605.00 | 279.01 | 632.90 |
| 1.0 | 656.34 | 124.01 | 780.35 |
| 5.0 | 821.32 | 56.86 | 1105.63 |

Lecture : quand beta augmente, la KL est davantage penalisee, donc elle baisse (latent plus proche de la loi normale, mieux regularise), mais la reconstruction se degrade (image reconstruite moins fidele). Un beta trop petit fait l'inverse : bonne reconstruction, mais latent peu structure, avec un risque de mauvaise generation quand on echantillonne un z aleatoire.

La configuration actuelle prepare une ablation plus fine (`configs/ablation_beta.yaml`) avec les valeurs `[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`, 40 epochs maximum et arret anticipe. Quand elle sera relancee, ce fichier sera regenere avec la meilleure validation atteinte pour chaque beta.
