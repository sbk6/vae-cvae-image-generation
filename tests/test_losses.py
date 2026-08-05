import torch

from src.losses.elbo import elbo_loss


def test_kl_zero_for_standard_normal() -> None:
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    recon = torch.zeros(4, 1, 28, 28)
    x = torch.zeros(4, 1, 28, 28)
    metrics = elbo_loss(recon, x, mu, logvar, beta=1.0)
    assert torch.isclose(metrics["kl"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(metrics["loss"], torch.tensor(0.0), atol=1e-6)
