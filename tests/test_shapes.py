import torch

from src.models.vae import VAE


def test_vae_output_shapes() -> None:
    model = VAE(channels=1, image_size=(28, 28), latent_dim=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
