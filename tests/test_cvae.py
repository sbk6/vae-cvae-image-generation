import torch

from src.models.cvae import CVAE


def test_cvae_shapes() -> None:
    model = CVAE(channels=1, image_size=(28, 28), latent_dim=8, num_conditions=10, condition_mode='one_hot', hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    c = torch.tensor([0, 1, 2, 3])
    x_hat, mu, logvar = model(x, c)
    assert x_hat.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_cvae_sample_reconstruct() -> None:
    model = CVAE(channels=1, image_size=(28, 28), latent_dim=8, num_conditions=10, condition_mode='one_hot', hidden_channels=16)
    sample = model.sample(2, n=5)
    assert sample.shape == (5, 1, 28, 28)
