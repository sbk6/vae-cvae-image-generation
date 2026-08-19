"""Agrège les résultats de l'étude d'ablation multi-seed (3 betas x 3 seeds, 20 epochs).

Lit `training_log.csv` de chaque run `reports/experiments/ablation_seeds/beta_{b}_seed_{s}/`,
calcule la moyenne et l'écart-type sur les seeds pour chaque beta, et produit :
- un tableau Markdown (ajouté à docs/RESULTATS.md)
- une courbe avec barres d'erreur (reports/figures/ablation_beta_seeds_curve.png)
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from src.utils.config import load_yaml_config


def read_final_val(output_dir: Path) -> dict:
    log_path = output_dir / "training_log.csv"
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["phase"] == "val"]
    last = rows[-1]
    return {"loss": float(last["loss"]), "reconstruction": float(last["reconstruction"]), "kl": float(last["kl"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablation_beta.yaml")
    parser.add_argument("--results-md", type=str, default="docs/RESULTATS.md")
    parser.add_argument("--figure", type=str, default="reports/figures/ablation_beta_seeds_curve.png")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    betas = config["ablation"]["betas"]
    seeds = config["ablation"]["seeds"]
    root = Path(config["training"]["output_dir"])

    per_beta = {}
    for beta in betas:
        runs = []
        for seed in seeds:
            output_dir = root / f"beta_{beta}_seed_{seed}"
            if not (output_dir / "training_log.csv").exists():
                print(f"[manquant] {output_dir}")
                continue
            runs.append(read_final_val(output_dir))
        per_beta[beta] = runs

    lines = []
    lines.append("\n# Ablation multi-seed (3 seeds x 20 epochs)\n")
    lines.append(
        f"Protocole : VAE identique (latent_dim={config['model']['latent_dim']}, "
        f"hidden_channels={config['model']['hidden_channels']}, epochs={config['training']['epochs']}, "
        f"train_subset={config['dataset'].get('train_subset')}), chaque beta répété avec les seeds {seeds}.\n"
    )
    lines.append("| beta | reconstruction (moyenne ± écart-type) | KL (moyenne ± écart-type) | loss totale (moyenne ± écart-type) |")
    lines.append("|---|---|---|---|")

    summary = {}
    for beta in betas:
        runs = per_beta[beta]
        if not runs:
            continue
        recon = [r["reconstruction"] for r in runs]
        kl = [r["kl"] for r in runs]
        loss = [r["loss"] for r in runs]

        def fmt(values):
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean, std, f"{mean:.2f} ± {std:.2f}"

        recon_mean, recon_std, recon_fmt = fmt(recon)
        kl_mean, kl_std, kl_fmt = fmt(kl)
        loss_mean, loss_std, loss_fmt = fmt(loss)
        summary[beta] = {"recon_mean": recon_mean, "recon_std": recon_std, "kl_mean": kl_mean, "kl_std": kl_std}
        lines.append(f"| {beta} | {recon_fmt} | {kl_fmt} | {loss_fmt} | (n={len(runs)} seeds)")

    lines.append("")
    lines.append(
        "Lecture : les écarts-types indiquent la variabilité d'un run à l'autre pour un même beta, "
        "uniquement due au seed (initialisation des poids + ordre des batches). Un écart-type petit "
        "par rapport à l'écart entre les betas confirme que l'effet observé vient bien de beta, pas du hasard."
    )

    with open(args.results_md, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    betas_sorted = sorted(summary.keys())
    recon_means = [summary[b]["recon_mean"] for b in betas_sorted]
    recon_stds = [summary[b]["recon_std"] for b in betas_sorted]
    kl_means = [summary[b]["kl_mean"] for b in betas_sorted]
    kl_stds = [summary[b]["kl_std"] for b in betas_sorted]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("beta")
    ax1.set_ylabel("reconstruction (val)", color=color1)
    ax1.errorbar(betas_sorted, recon_means, yerr=recon_stds, marker="o", color=color1, capsize=4, label="reconstruction")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("KL (val)", color=color2)
    ax2.errorbar(betas_sorted, kl_means, yerr=kl_stds, marker="s", color=color2, capsize=4, label="KL")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"Effet de beta (moyenne ± écart-type sur {len(seeds)} seeds, 20 epochs)")
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, bbox_inches="tight")
    plt.close(fig)

    print(f"Résultats ajoutés à {args.results_md}, figure sauvegardée dans {args.figure}")


if __name__ == "__main__":
    main()
