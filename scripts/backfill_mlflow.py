"""Réimporte dans MLflow les entraînements déjà réalisés avant que le tracking
MLflow ne soit branché dans `src/training/trainer.py`.

On ne réentraîne rien : on relit `training_log.csv` et `best_checkpoint.pth`
de chaque expérience déjà exécutée (`reports/experiments/...`) et on les
rejoue dans MLflow (mêmes métriques, épreuve par époque). Les prochains
entraînements seront eux logués en direct par `train()`.
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src.utils.config import load_yaml_config

EXPERIMENTS = [
    {
        "config_path": "configs/mnist_vae.yaml",
        "output_dir": "reports/experiments/vae_main",
        "run_name": "vae_main",
        "tags": {"phase": "main", "model": "vae", "backfilled": "true"},
        "overrides": {},
    },
    {
        "config_path": "configs/mnist_cvae.yaml",
        "output_dir": "reports/experiments/cvae_main",
        "run_name": "cvae_main",
        "tags": {"phase": "main", "model": "cvae", "backfilled": "true"},
        "overrides": {},
    },
]

for beta in [0.1, 1.0, 5.0]:
    EXPERIMENTS.append(
        {
            "config_path": "configs/ablation_beta.yaml",
            "output_dir": f"reports/experiments/ablation/beta_{beta}",
            "run_name": f"ablation_beta_{beta}",
            "tags": {"phase": "ablation", "beta": str(beta), "backfilled": "true"},
            "overrides": {"beta": beta},
            "experiment_name": "vae-cvae-mnist-ablation",
        }
    )


def backfill_one(spec: dict, tracking_uri: str) -> None:
    config = load_yaml_config(spec["config_path"])
    for key, value in spec["overrides"].items():
        config["training"][key] = value

    output_dir = Path(spec["output_dir"])
    log_path = output_dir / "training_log.csv"
    ckpt_path = output_dir / "best_checkpoint.pth"
    if not log_path.exists():
        print(f"[skip] {log_path} introuvable")
        return

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = spec.get("experiment_name", config["training"].get("mlflow", {}).get("experiment_name", "vae-cvae-mnist"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=spec["run_name"]) as run:
        mlflow.set_tags(spec["tags"])
        mlflow.log_params(
            {
                "model_type": config.get("model", {}).get("type", "vae"),
                "dataset": config.get("dataset", {}).get("name"),
                "train_subset": config.get("dataset", {}).get("train_subset", "full"),
                "latent_dim": config.get("model", {}).get("latent_dim"),
                "hidden_channels": config.get("model", {}).get("hidden_channels"),
                "beta": config["training"].get("beta", 1.0),
                "lr": config["training"]["lr"],
                "batch_size": config["training"]["batch_size"],
                "epochs": config["training"].get("epochs"),
                "seed": config["training"].get("seed"),
            }
        )

        best_val_loss = float("inf")
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            per_epoch = {}
            for row in reader:
                epoch = int(row["epoch"])
                per_epoch.setdefault(epoch, {})[row["phase"]] = row

        for epoch in sorted(per_epoch.keys()):
            metrics = {}
            for phase in ("train", "val"):
                row = per_epoch[epoch].get(phase)
                if row is None:
                    continue
                for key in ("loss", "reconstruction", "kl"):
                    metrics[f"{phase}_{key}"] = float(row[key])
                if phase == "val":
                    best_val_loss = min(best_val_loss, float(row["loss"]))
            mlflow.log_metrics(metrics, step=epoch)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(str(log_path))
        if ckpt_path.exists():
            mlflow.log_artifact(str(ckpt_path))

        print(f"[ok] {spec['run_name']} -> run_id={run.info.run_id}, best_val_loss={best_val_loss:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    args = parser.parse_args()

    for spec in EXPERIMENTS:
        backfill_one(spec, args.tracking_uri)


if __name__ == "__main__":
    main()
