# Résultats — Étude d'ablation sur beta (beta-VAE)

Protocole : VAE identique pour toutes les valeurs de beta (latent_dim=16, hidden_channels=32, epochs=6, seed=42, train_subset=12000).

| beta | reconstruction (val) | KL (val) | loss totale (val) |
|---|---|---|---|
| 0.1 | 680.87 | 39.47 | 684.82 |
| 1.0 | 691.83 | 15.37 | 707.20 |
| 5.0 | 725.11 | 0.56 | 727.90 |

Lecture du tableau : quand `beta` augmente, la KL est davantage pénalisée, donc elle a tendance à baisser (latent plus proche de la loi normale, mieux régularisé), mais la reconstruction se dégrade (l'image reconstruite est moins fidèle). Un `beta` trop petit fait l'inverse : bonne reconstruction mais latent peu structuré, avec un risque de mauvaise génération quand on échantillonne un `z` aléatoire.
