"""Évaluation quantitative finale : VAE vs CVAE.

1) Reconstruction / KL sur le test set complet (déterministe, avec mu).
2) Contrôlabilité du CVAE : pour chaque classe demandée, on génère des
   échantillons et on vérifie s'ils tombent bien dans la bonne classe.
   Faute de classifieur pré-entraîné disponible hors-ligne, on utilise un
   classifieur "plus proche centroïde" : le centroïde de chaque classe est
   la moyenne des images réelles du train set pour cette classe (en espace
   pixel normalisé). Un échantillon généré est jugé "correct" s'il est plus
   proche du centroïde de la classe demandée que de tout autre centroïde.
   C'est une mesure simple et rapide (pas d'entraînement supplémentaire),
   utilisée ici comme proxy de contrôlabilité, pas comme un FID.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.data.datasets import build_dataloaders
from src.training.trainer import get_device
from src.utils.config import load_yaml_config
from src.visualization.common import build_model_from_config, load_checkpoint


@torch.no_grad()
def compute_recon_kl(config_path: str, checkpoint_path: str) -> dict:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    from src.losses.elbo import elbo_loss

    total_recon, total_kl, total = 0.0, 0.0, 0
    for x, y in test_loader:
        x = x.to(device)
        if conditioned:
            x_hat, mu, logvar = model(x, y.to(device))
        else:
            x_hat, mu, logvar = model(x)
        m = elbo_loss(x_hat, x, mu, logvar, beta=1.0)
        total_recon += m["reconstruction"].item() * x.size(0)
        total_kl += m["kl"].item() * x.size(0)
        total += x.size(0)

    return {"reconstruction": total_recon / total, "kl": total_kl / total, "n_test": total}


@torch.no_grad()
def compute_class_centroids(config_path: str) -> torch.Tensor:
    config = load_yaml_config(config_path)
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)
    num_classes = dataset_info.num_conditions
    sums = torch.zeros(num_classes, dataset_info.channels, *dataset_info.image_size)
    counts = torch.zeros(num_classes)
    for x, y in train_loader:
        for i in range(x.size(0)):
            sums[y[i]] += x[i]
            counts[y[i]] += 1
    centroids = sums / counts.view(-1, 1, 1, 1)
    return centroids


@torch.no_grad()
def evaluate_controllability(config_path: str, checkpoint_path: str, centroids: torch.Tensor, n_per_class: int = 200) -> dict:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    centroids_flat = centroids.view(centroids.size(0), -1).to(device)
    per_class_accuracy = {}
    total_correct, total = 0, 0
    for cls in range(dataset_info.num_conditions):
        samples = model.sample(cls, n=n_per_class).to(device)
        flat = samples.view(samples.size(0), -1)
        dists = torch.cdist(flat, centroids_flat)
        predicted = dists.argmin(dim=1)
        correct = (predicted == cls).sum().item()
        per_class_accuracy[str(cls)] = correct / n_per_class
        total_correct += correct
        total += n_per_class

    return {"per_class_accuracy": per_class_accuracy, "overall_accuracy": total_correct / total}


@torch.no_grad()
def evaluate_vae_diversity(config_path: str, checkpoint_path: str, centroids: torch.Tensor, n_samples: int = 1000) -> dict:
    """Pour le VAE non conditionnel : on échantillonne librement puis on regarde
    la répartition des classes prédites par plus-proche-centroïde. Une bonne
    couverture de la distribution donne une répartition proche de l'uniforme
    (10% par classe pour MNIST)."""
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    z = torch.randn(n_samples, model.latent_dim, device=device)
    samples = model.decode(z)
    centroids_flat = centroids.view(centroids.size(0), -1).to(device)
    flat = samples.view(samples.size(0), -1)
    dists = torch.cdist(flat, centroids_flat)
    predicted = dists.argmin(dim=1)
    counts = torch.bincount(predicted, minlength=dataset_info.num_conditions).float()
    distribution = (counts / n_samples).tolist()
    return {"predicted_class_distribution": distribution, "n_samples": n_samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-config", type=str, default="configs/mnist_vae.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="reports/experiments/vae_main/best_checkpoint.pth")
    parser.add_argument("--cvae-config", type=str, default="configs/mnist_cvae.yaml")
    parser.add_argument("--cvae-checkpoint", type=str, default="reports/experiments/cvae_main/best_checkpoint.pth")
    parser.add_argument("--output", type=str, default="reports/experiments/comparison.json")
    args = parser.parse_args()

    print("Calcul reconstruction/KL sur le test set (VAE)...")
    vae_metrics = compute_recon_kl(args.vae_config, args.vae_checkpoint)
    print(vae_metrics)

    print("Calcul reconstruction/KL sur le test set (CVAE)...")
    cvae_metrics = compute_recon_kl(args.cvae_config, args.cvae_checkpoint)
    print(cvae_metrics)

    print("Calcul des centroïdes de classe (train set réel)...")
    centroids = compute_class_centroids(args.vae_config)

    print("Évaluation de la contrôlabilité du CVAE (plus-proche-centroïde)...")
    controllability = evaluate_controllability(args.cvae_config, args.cvae_checkpoint, centroids)
    print(controllability)

    print("Évaluation de la couverture des classes pour le VAE (génération libre)...")
    vae_diversity = evaluate_vae_diversity(args.vae_config, args.vae_checkpoint, centroids)
    print(vae_diversity)

    results = {
        "vae": {**vae_metrics, "generation_class_distribution": vae_diversity},
        "cvae": {**cvae_metrics, "controllability": controllability},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()
