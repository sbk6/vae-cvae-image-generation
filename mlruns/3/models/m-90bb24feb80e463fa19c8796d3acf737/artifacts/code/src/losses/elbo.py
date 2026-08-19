import torch
import torch.nn.functional as F


def elbo_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict:
    """Calcule la loss ELBO pour le VAE.

    La reconstruction est mesurée par l'erreur quadratique moyenne
    en espace d'image normalisé. La divergence KL est calculée
    analytiquement entre la distribution q(z|x)=N(mu, sigma^2) et la
    prior p(z)=N(0, I).

    La formule KL repose sur la forme analytique de la divergence entre
    deux lois normales multivariées diagonales.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_loss + beta * kl_div
    return {
        "loss": loss,
        "reconstruction": recon_loss,
        "kl": kl_div,
        "beta": beta,
    }
