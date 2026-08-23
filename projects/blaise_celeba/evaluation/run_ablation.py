"""Etude d'ablation sur le poids beta de la loss ELBO (beta-VAE), sur CelebA.

Pour chaque valeur de beta listee dans configs/ablation_beta.yaml, entraine un
VAE depuis zero (meme seed, meme architecture, meme sous-echantillon, meme
budget maximal d'epochs) et sauvegarde les meilleures metriques de validation.
Genere un tableau Markdown et une courbe reconstruction/KL en fonction de beta.

Structure identique a scripts/run_ablation.py (Sylvain, MNIST), pour que les
deux etudes d'ablation se comparent directement dans le rapport final.

Usage (depuis projects/blaise_celeba/) :
    python -m evaluation.run_ablation --config configs/ablation_beta.yaml
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from data.dataset import build_dataloaders
from models.vae import VAE
from training.trainer import train
from utils import load_yaml_config, set_seed


def run_one_beta(base_config: dict, beta: float) -> dict:
    config = copy.deepcopy(base_config)
    config["training"]["beta"] = beta
    config["training"]["output_dir"] = str(Path(base_config["training"]["output_dir"]) / f"beta_{beta}")
    config["smoke_test"] = False

    set_seed(config["training"].get("seed", 42))
    train_loader, val_loader, test_loader, dataset_info = build_dataloaders(config)

    model = VAE(
        channels=dataset_info.channels,
        image_size=dataset_info.image_size,
        latent_dim=config["model"]["latent_dim"],
        hidden_channels=config["model"]["hidden_channels"],
    )

    train(config, model, train_loader, val_loader)

    log_path = Path(config["training"]["output_dir"]) / "training_log.csv"
    best_val = None
    with open(log_path, "r", encoding="utf-8") as log_file:
        reader = csv.DictReader(log_file)
        for row in reader:
            if row["phase"] == "val":
                if best_val is None or float(row["loss"]) < float(best_val["loss"]):
                    best_val = row

    return {
        "beta": beta,
        "best_epoch": int(best_val["epoch"]),
        "best_val_loss": float(best_val["loss"]),
        "best_val_reconstruction": float(best_val["reconstruction"]),
        "best_val_kl": float(best_val["kl"]),
        "output_dir": config["training"]["output_dir"],
    }


def write_markdown_table(results: list, output_path: str, base_config: dict) -> None:
    lines = [
        "# Resultats. Etude d'ablation sur beta (beta-VAE), CelebA\n",
        (
            f"Protocole : VAE identique pour toutes les valeurs de beta "
            f"(latent_dim={base_config['model']['latent_dim']}, "
            f"hidden_channels={base_config['model']['hidden_channels']}, "
            f"epochs={base_config['training']['epochs']}, "
            f"seed={base_config['training']['seed']}, "
            f"n_train={base_config['dataset']['n_train']}).\n"
        ),
        "| beta | meilleure epoch | reconstruction (val) | KL (val) | loss totale (val) |",
        "|---|---|---|---|---|",
    ]
    for result in sorted(results, key=lambda item: item["beta"]):
        lines.append(
            f"| {result['beta']} | {result['best_epoch']} | {result['best_val_reconstruction']:.2f} | "
            f"{result['best_val_kl']:.2f} | {result['best_val_loss']:.2f} |"
        )
    lines.append("")
    lines.append(
        "Lecture : quand beta augmente, la KL est davantage penalisee, donc elle "
        "baisse (latent plus proche de la loi normale, mieux regularise), mais la "
        "reconstruction se degrade (image reconstruite moins fidele). Un beta trop "
        "petit fait l'inverse : bonne reconstruction, mais latent peu structure, "
        "avec un risque de mauvaise generation quand on echantillonne un z aleatoire."
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write("\n".join(lines) + "\n")


def plot_curve(results: list, output_path: str) -> None:
    results_sorted = sorted(results, key=lambda item: item["beta"])
    betas = [item["beta"] for item in results_sorted]
    recon = [item["best_val_reconstruction"] for item in results_sorted]
    kl = [item["best_val_kl"] for item in results_sorted]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("beta")
    ax1.set_ylabel("reconstruction (val)", color=color1)
    ax1.plot(betas, recon, marker="o", color=color1, label="reconstruction")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("KL (val)", color=color2)
    ax2.plot(betas, kl, marker="s", color=color2, label="KL")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("CelebA : effet de beta sur reconstruction vs KL")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ablation_beta.yaml")
    parser.add_argument("--results-json", type=str, default="results/experiments/ablation/results.json")
    parser.add_argument("--results-md", type=str, default="results/RESULTATS.md")
    parser.add_argument("--figure", type=str, default="results/figures/ablation_beta_curve.png")
    args = parser.parse_args()

    base_config = load_yaml_config(args.config)
    betas = base_config["ablation"]["betas"]

    results = []
    for beta in betas:
        print(f"=== Ablation CelebA : beta={beta} ===")
        result = run_one_beta(base_config, beta)
        results.append(result)
        print(result)

    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    write_markdown_table(results, args.results_md, base_config)
    plot_curve(results, args.figure)
    print(f"Resultats sauvegardes dans {args.results_json}, {args.results_md}, {args.figure}")


if __name__ == "__main__":
    main()
