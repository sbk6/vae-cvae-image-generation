"""Adaptateurs par famille de modeles.

Chaque module expose une fonction `load(checkpoint_path, device, **kwargs)`
renvoyant un `ModelAdapter`. Le catalogue designe l'adaptateur a utiliser par
son nom, ce qui evite au reste du backend d'importer les modeles directement.
"""
from backend.adapters.base import ModelAdapter
from backend.adapters import blaise_celeba, david_fashion, sylvain_mnist

LOADERS = {
    "sylvain_mnist": sylvain_mnist.load,
    "david_fashion": david_fashion.load,
    "blaise_celeba": blaise_celeba.load,
}

__all__ = ["ModelAdapter", "LOADERS", "sylvain_mnist", "david_fashion", "blaise_celeba"]
