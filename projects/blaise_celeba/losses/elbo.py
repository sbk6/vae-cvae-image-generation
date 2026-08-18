"""Perte ELBO (Evidence Lower BOund) utilisee pour entrainer le VAE et le CVAE.

    loss = reconstruction + beta * KL

- reconstruction : erreur quadratique (MSE), sommee sur les pixels et
  canaux d'une image puis moyennee sur le batch. Correspond a un decodeur
  gaussien a variance fixe dans la derivation de l'ELBO : hypothese adaptee a
  des intensites de pixel continues comme des photos couleur. C'est un choix
  different de l'implementation Fashion-MNIST de l'equipe, qui utilise la BCE
  (decodeur de Bernoulli, plus adaptee a des pixels quasi binaires). Les deux
  sont documentees comme des choix assumes, pas des oublis.
- KL : divergence de Kullback-Leibler entre la distribution latente apprise
  q(z|x) = N(mu, diag(exp(logvar))) et le prior N(0, I). Calculee sous forme
  analytique fermee, valable pour deux gaussiennes diagonales. Elle force
  l'espace latent a rester proche d'une gaussienne standard, condition
  necessaire pour pouvoir ensuite generer une image en tirant z ~ N(0, I).
- beta : poids applique au terme KL (beta-VAE, Higgins et al. 2017). C'est le
  parametre etudie dans l'etude d'ablation (evaluation/run_ablation.py).

Convention de normalisation (somme sur les pixels/dimensions latentes, puis
moyenne sur le batch) identique a celle de src/losses/elbo.py (Sylvain,
MNIST), pour que les valeurs de reconstruction/KL des deux sous-projets
restent comparables dans le rapport final, meme si les images n'ont pas la
meme taille.
"""
from typing import Dict

import torch
import torch.nn.functional as F


def elbo_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_loss + beta * kl_div
    return {
        "loss": loss,
        "reconstruction": recon_loss,
        "kl": kl_div,
        "beta": beta,
    }
