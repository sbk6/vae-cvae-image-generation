import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

from src.losses.elbo import elbo_loss


@dataclass
class TrainingState:
    epoch: int
    best_loss: float
    device: torch.device
    output_dir: Path


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: TrainingState) -> None:
    state_path = state.output_dir / "best_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": state.epoch,
            "best_loss": state.best_loss,
        },
        state_path,
    )


def initialize_csv(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "phase", "loss", "reconstruction", "kl", "beta"])
            writer.writeheader()
    return csv_path


def log_metrics(csv_path: Path, epoch: int, phase: str, metrics: Dict[str, float]) -> None:
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "phase", "loss", "reconstruction", "kl", "beta"])
        writer.writerow({"epoch": epoch, "phase": phase, **metrics})


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    beta: float,
    is_train: bool = True,
    smoke_test: bool = False,
    conditioned: bool = False,
) -> Dict[str, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0

    iterator = tqdm(dataloader, desc="train" if is_train else "val", unit="batch")
    for batch_idx, batch in enumerate(iterator):
        # expect dataset to yield (x, y) where y is label or condition vector
        x, y = batch
        x = x.to(device)

        if conditioned:
            c = y.to(device)
        else:
            c = None

        if is_train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if conditioned:
                x_hat, mu, logvar = model(x, c)
            else:
                x_hat, mu, logvar = model(x)
            metrics = elbo_loss(x_hat, x, mu, logvar, beta=beta)
            if is_train and optimizer is not None:
                metrics["loss"].backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += metrics["loss"].item() * batch_size
        total_recon += metrics["reconstruction"].item() * batch_size
        total_kl += metrics["kl"].item() * batch_size
        total_samples += batch_size

        if smoke_test and batch_idx >= 49:
            break

    averaged = {
        "loss": total_loss / total_samples,
        "reconstruction": total_recon / total_samples,
        "kl": total_kl / total_samples,
        "beta": beta,
    }
    return averaged


def start_mlflow_run(config: Dict):
    """Démarre (ou non) un run MLflow selon `training.mlflow` dans le YAML.

    Retourne un context manager : soit un vrai run MLflow, soit un
    `contextlib.nullcontext()` si MLflow est désactivé. Le reste du code
    n'a donc pas besoin de savoir si MLflow est actif ou non.
    """
    import contextlib

    mlflow_cfg = config["training"].get("mlflow", {})
    if not mlflow_cfg.get("enabled", False):
        return contextlib.nullcontext(), False

    import mlflow

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "vae-cvae-mnist"))
    run = mlflow.start_run(run_name=mlflow_cfg.get("run_name"))
    tags = mlflow_cfg.get("tags", {})
    if tags:
        mlflow.set_tags(tags)
    return run, True


def log_mlflow_params(config: Dict, model: torch.nn.Module) -> None:
    import mlflow

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
            "epochs": config["training"].get("epochs", 10),
            "seed": config["training"].get("seed"),
            "n_parameters": sum(p.numel() for p in model.parameters()),
        }
    )


def train(
    config: Dict,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> None:
    device = get_device(config["training"].get("device", "auto"))
    output_dir = Path(config["training"].get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    state = TrainingState(epoch=0, best_loss=float("inf"), device=device, output_dir=output_dir)
    csv_path = initialize_csv(output_dir)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    beta = config["training"].get("beta", 1.0)
    epochs = config["training"].get("epochs", 10)
    smoke_test = config.get("smoke_test", False)
    if smoke_test:
        epochs = 1

    # detect whether model expects a condition argument (CVAE)
    conditioned = getattr(model, "num_conditions", None) is not None

    run_ctx, use_mlflow = start_mlflow_run(config)
    with run_ctx:
        if use_mlflow:
            import mlflow

            log_mlflow_params(config, model)

        for epoch in range(1, epochs + 1):
            state.epoch = epoch
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, beta, is_train=True, smoke_test=smoke_test, conditioned=conditioned
            )
            val_metrics = run_epoch(
                model, val_loader, None, device, beta, is_train=False, smoke_test=smoke_test, conditioned=conditioned
            )

            log_metrics(csv_path, epoch, "train", train_metrics)
            log_metrics(csv_path, epoch, "val", val_metrics)

            if use_mlflow:
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items() if k != "beta"}, step=epoch)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if k != "beta"}, step=epoch)

            if val_metrics["loss"] < state.best_loss:
                state.best_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, state)

        if use_mlflow:
            mlflow.log_metric("best_val_loss", state.best_loss)
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(output_dir / "best_checkpoint.pth"))

    print(f"Training completed. Best validation loss: {state.best_loss:.4f}")
