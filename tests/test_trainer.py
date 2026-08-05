import torch

from src.models.vae import VAE
from src.models.cvae import CVAE
from src.training.trainer import run_epoch


def test_run_epoch_shape() -> None:
    model = VAE(channels=1, image_size=(28, 28), latent_dim=8, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    dataset = [(x[i], torch.tensor(i)) for i in range(4)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    metrics = run_epoch(model, loader, None, torch.device("cpu"), beta=1.0, is_train=False)
    assert "loss" in metrics
    assert "reconstruction" in metrics
    assert "kl" in metrics


def test_run_epoch_conditioned_shape() -> None:
    model = CVAE(channels=1, image_size=(28, 28), latent_dim=8, num_conditions=10, condition_mode='one_hot', hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    c = torch.tensor([0, 1, 2, 3])
    dataset = [(x[i], c[i]) for i in range(4)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    metrics = run_epoch(model, loader, None, torch.device("cpu"), beta=1.0, is_train=False, conditioned=True)
    assert "loss" in metrics
    assert "reconstruction" in metrics
    assert "kl" in metrics
