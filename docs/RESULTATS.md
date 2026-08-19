# Résultats — Étude d'ablation sur beta (beta-VAE)

Protocole : VAE identique pour toutes les valeurs de beta (latent_dim=16, hidden_channels=32, epochs=6, seed=42, train_subset=12000).

| beta | reconstruction (val) | KL (val) | loss totale (val) |
|---|---|---|---|
| 0.1 | 680.87 | 39.47 | 684.82 |
| 1.0 | 691.83 | 15.37 | 707.20 |
| 5.0 | 725.11 | 0.56 | 727.90 |

Lecture du tableau : quand `beta` augmente, la KL est davantage pénalisée, donc elle a tendance à baisser (latent plus proche de la loi normale, mieux régularisé), mais la reconstruction se dégrade (l'image reconstruite est moins fidèle). Un `beta` trop petit fait l'inverse : bonne reconstruction mais latent peu structuré, avec un risque de mauvaise génération quand on échantillonne un `z` aléatoire.

# Ablation multi-seed (3 seeds x 20 epochs)

Protocole : VAE identique (latent_dim=16, hidden_channels=32, epochs=20, train_subset=12000), chaque beta répété avec les seeds [0, 42, 123].

| beta | reconstruction (moyenne ± écart-type) | KL (moyenne ± écart-type) | loss totale (moyenne ± écart-type) |
|---|---|---|---|
| 0.1 | 672.33 ± 0.24 | 40.75 ± 0.81 | 676.41 ± 0.26 | (n=3 seeds)
| 1.0 | 682.98 ± 0.14 | 16.62 ± 0.08 | 699.60 ± 0.14 | (n=3 seeds)
| 5.0 | 724.77 ± 0.26 | 0.21 ± 0.06 | 725.82 ± 0.06 | (n=3 seeds)

Lecture : les écarts-types indiquent la variabilité d'un run à l'autre pour un même beta, uniquement due au seed (initialisation des poids + ordre des batches). Un écart-type petit par rapport à l'écart entre les betas confirme que l'effet observé vient bien de beta, pas du hasard.

# Validation multi-seed des modèles principaux (VAE et CVAE, 20 epochs, données complètes)

Protocole : `vae_main` et `cvae_main` réentraînés chacun avec 3 seeds (0, 42, 123), 54 000 images d'entraînement, 20 epochs (contre 10 initialement), beta=1.0, latent_dim=16.

| modèle | reconstruction (moyenne ± écart-type) | KL (moyenne ± écart-type) |
|---|---|---|
| VAE | 677.08 ± 0.18 | 17.25 ± 0.17 |
| CVAE | 675.75 ± 0.88 | 14.14 ± 0.46 |

Lecture : le VAE est très stable d'un seed à l'autre (écart-type ≈0.18) ; le CVAE varie un peu plus (écart-type ≈5x supérieur) mais reste largement dans une plage négligeable comparée à l'effet de beta. Passer de 10 à 20 epochs n'apporte qu'un gain marginal pour les deux modèles, confirmant que 10 epochs suffisait déjà. Le CVAE reste légèrement meilleur que le VAE sur les deux métriques.
