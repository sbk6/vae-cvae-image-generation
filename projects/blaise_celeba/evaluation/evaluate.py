"""Evaluation quantitative finale : VAE vs CVAE sur CelebA.

1) Reconstruction / KL sur le test set (deterministe, avec mu).
2) Controlabilite du CVAE : pour chaque attribut de conditionnement, on
   calcule un centroide "attribut actif" et un centroide "attribut inactif"
   a partir des vraies images d'entrainement (moyenne pixel par pixel). On
   genere ensuite des images en demandant chaque combinaison possible du
   vecteur multi-hot, et on verifie, attribut par attribut, si l'image
   generee est plus proche du centroide "actif" ou "inactif" attendu.

   Meme principe que le proxy "plus proche centroide" utilise par Sylvain sur
   MNIST (scripts/evaluate.py), adapte au cas multi-label : au lieu d'une
   seule prediction de classe parmi 10, on fait une prediction binaire par
   attribut, independamment pour chacun des 3 attributs. Faute de
   classifieur pre-entraine disponible hors ligne, c'est un proxy simple et
   rapide, pas un score de type FID : ses limites (sensibilite au flou des
   images generees par un VAE) sont documentees dans le README, dans les
   memes termes que pour MNIST.

Usage (depuis projects/blaise_celeba/) :
    python -m evaluation.evaluate
"""
import argparse
import itertools
import json
from pathlib import Path

import torch

from data.dataset import build_dataloaders
from losses.elbo import elbo_loss
from models.common import build_model_from_config, load_checkpoint
from training.trainer import get_device
from utils import load_yaml_config


@torch.no_grad()
def compute_recon_kl(config_path: str, checkpoint_path: str) -> dict:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"

    total_recon = total_kl = 0.0
    total = 0
    for x, attrs in test_loader:
        x = x.to(device)
        if conditioned:
            x_hat, mu, logvar = model(x, attrs.to(device).float())
        else:
            x_hat, mu, logvar = model(x)
        metrics = elbo_loss(x_hat, x, mu, logvar, beta=1.0)
        total_recon += metrics["reconstruction"].item() * x.size(0)
        total_kl += metrics["kl"].item() * x.size(0)
        total += x.size(0)

    return {"reconstruction": total_recon / total, "kl": total_kl / total, "n_test": total}


@torch.no_grad()
def compute_attribute_centroids(config_path: str) -> dict:
    """Pour chaque attribut : image moyenne des exemples ou il est actif, et ou il ne l'est pas."""
    config = load_yaml_config(config_path)
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    num_attrs = dataset_info.num_conditions
    channels, height, width = dataset_info.channels, *dataset_info.image_size
    sums = {"active": torch.zeros(num_attrs, channels, height, width), "inactive": torch.zeros(num_attrs, channels, height, width)}
    counts = {"active": torch.zeros(num_attrs), "inactive": torch.zeros(num_attrs)}

    for x, attrs in train_loader:
        for attr_index in range(num_attrs):
            active_mask = attrs[:, attr_index] > 0.5
            sums["active"][attr_index] += x[active_mask].sum(dim=0)
            counts["active"][attr_index] += active_mask.sum()
            sums["inactive"][attr_index] += x[~active_mask].sum(dim=0)
            counts["inactive"][attr_index] += (~active_mask).sum()

    centroids = {
        "active": sums["active"] / counts["active"].clamp(min=1).view(-1, 1, 1, 1),
        "inactive": sums["inactive"] / counts["inactive"].clamp(min=1).view(-1, 1, 1, 1),
    }
    return centroids


@torch.no_grad()
def evaluate_controllability(config_path: str, checkpoint_path: str, centroids: dict, n_per_combo: int = 100) -> dict:
    config = load_yaml_config(config_path)
    device = get_device(config["training"].get("device", "auto"))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = build_model_from_config(config, dataset_info)
    model = load_checkpoint(model, checkpoint_path, device)

    num_attrs = dataset_info.num_conditions
    active_flat = centroids["active"].view(num_attrs, -1).to(device)
    inactive_flat = centroids["inactive"].view(num_attrs, -1).to(device)

    per_attribute_correct = torch.zeros(num_attrs)
    per_attribute_total = torch.zeros(num_attrs)

    for combo in itertools.product([0.0, 1.0], repeat=num_attrs):
        condition = torch.tensor(combo, dtype=torch.float32)
        samples = model.sample(condition, n=n_per_combo).to(device)
        flat = samples.view(samples.size(0), -1)
        for attr_index in range(num_attrs):
            requested = combo[attr_index]
            dist_active = torch.norm(flat - active_flat[attr_index], dim=1)
            dist_inactive = torch.norm(flat - inactive_flat[attr_index], dim=1)
            predicted_active = (dist_active < dist_inactive).float()
            expected = torch.full_like(predicted_active, requested)
            per_attribute_correct[attr_index] += (predicted_active == expected).sum().item()
            per_attribute_total[attr_index] += n_per_combo

    per_attribute_accuracy = {
        name: (per_attribute_correct[i] / per_attribute_total[i]).item()
        for i, name in enumerate(dataset_info.attribute_names)
    }
    overall_accuracy = (per_attribute_correct.sum() / per_attribute_total.sum()).item()
    return {"per_attribute_accuracy": per_attribute_accuracy, "overall_accuracy": overall_accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-config", type=str, default="configs/celeba_vae.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="results/experiments/vae_main/best_checkpoint.pth")
    parser.add_argument("--cvae-config", type=str, default="configs/celeba_cvae.yaml")
    parser.add_argument("--cvae-checkpoint", type=str, default="results/experiments/cvae_main/best_checkpoint.pth")
    parser.add_argument("--output", type=str, default="results/experiments/comparison.json")
    args = parser.parse_args()

    print("Calcul reconstruction/KL sur le test set (VAE)...")
    vae_metrics = compute_recon_kl(args.vae_config, args.vae_checkpoint)
    print(vae_metrics)

    print("Calcul reconstruction/KL sur le test set (CVAE)...")
    cvae_metrics = compute_recon_kl(args.cvae_config, args.cvae_checkpoint)
    print(cvae_metrics)

    print("Calcul des centroides actif/inactif par attribut (train set reel)...")
    centroids = compute_attribute_centroids(args.cvae_config)

    print("Evaluation de la controlabilite du CVAE (plus-proche-centroide, par attribut)...")
    controllability = evaluate_controllability(args.cvae_config, args.cvae_checkpoint, centroids)
    print(controllability)

    results = {
        "vae": vae_metrics,
        "cvae": {**cvae_metrics, "controllability": controllability},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
    print(f"Resultats sauvegardes dans {args.output}")


if __name__ == "__main__":
    main()
