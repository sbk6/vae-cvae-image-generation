"""Boucle d'entrainement generique (VAE ou CVAE), avec journalisation CSV.

Structure volontairement proche de src/training/trainer.py (Sylvain) : meme
squelette epoch/validation/checkpoint/CSV, pour que les journaux
d'entrainement des deux sous-projets se lisent et se comparent de la meme
facon dans le rapport final. L'implementation reste independante (import de
losses.elbo, pas de src.losses.elbo).
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from tqdm import tqdm

from losses.elbo import elbo_loss
from mlflow_utils import MLflowLogger


@dataclass
class TrainingState:
    epoch: int
    best_loss: float
    best_epoch: int
    device: torch.device
    output_dir: Path


def beta_for_epoch(config: Dict, epoch: int) -> float:
    """Renvoie le beta a utiliser pour une epoch donnee.

    Par defaut, le beta est constant. Si `training.kl_annealing.enabled` est
    vrai, beta augmente lineairement de `start_beta` vers `final_beta` pendant
    `warmup_epochs`. C'est utile pour laisser le decodeur apprendre a
    reconstruire avant de regulariser fortement l'espace latent.
    """
    training_cfg = config["training"]
    final_beta = float(training_cfg.get("beta", 1.0))
    annealing_cfg = training_cfg.get("kl_annealing", {})
    if not annealing_cfg or not annealing_cfg.get("enabled", False):
        return final_beta

    start_beta = float(annealing_cfg.get("start_beta", 0.0))
    final_beta = float(annealing_cfg.get("final_beta", final_beta))
    warmup_epochs = max(1, int(annealing_cfg.get("warmup_epochs", 10)))
    progress = min(1.0, epoch / warmup_epochs)
    return start_beta + progress * (final_beta - start_beta)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: Dict,
    filename: str = "best_checkpoint.pth",
) -> None:
    state.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = state.output_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": state.epoch,
            "best_loss": state.best_loss,
            "best_epoch": state.best_epoch,
            "configuration": config,
        },
        state_path,
    )


def initialize_csv(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["epoch", "phase", "loss", "reconstruction", "kl", "beta"])
        writer.writeheader()
    return csv_path


def save_resolved_config(config: Dict, output_dir: Path) -> Path:
    """Sauvegarde la config effective du run, apres overrides CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False, allow_unicode=True)
    return config_path


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
    is_train: bool,
    conditioned: bool,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train() if is_train else model.eval()

    total_loss = total_recon = total_kl = 0.0
    total_samples = 0

    iterator = tqdm(dataloader, desc="train" if is_train else "val", unit="batch")
    for batch_idx, (x, attrs) in enumerate(iterator):
        x = x.to(device)

        if is_train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if conditioned:
                c = attrs.to(device).float()
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

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    return {
        "loss": total_loss / total_samples,
        "reconstruction": total_recon / total_samples,
        "kl": total_kl / total_samples,
        "beta": beta,
    }


def train(
    config: Dict,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> TrainingState:
    device = get_device(config["training"].get("device", "auto"))
    output_dir = Path(config["training"]["output_dir"])

    state = TrainingState(epoch=0, best_loss=float("inf"), best_epoch=0, device=device, output_dir=output_dir)
    csv_path = initialize_csv(output_dir)
    resolved_config_path = save_resolved_config(config, output_dir)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    epochs = config["training"].get("epochs", 15)
    conditioned = config.get("model", {}).get("type", "vae") == "cvae"
    early_stopping_cfg = config["training"].get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    patience = int(early_stopping_cfg.get("patience", 5))
    min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    epochs_without_improvement = 0

    smoke_test = config.get("smoke_test", False)
    max_batches = 5 if smoke_test else None
    if smoke_test:
        epochs = 1
        early_stopping_enabled = False

    with MLflowLogger(config) as mlflow_logger:
        mlflow_logger.log_artifact(resolved_config_path, artifact_path="config")
        for epoch in range(1, epochs + 1):
            state.epoch = epoch
            beta = beta_for_epoch(config, epoch)
            train_metrics = run_epoch(model, train_loader, optimizer, device, beta, True, conditioned, max_batches)
            val_metrics = run_epoch(model, val_loader, None, device, beta, False, conditioned, max_batches)

            log_metrics(csv_path, epoch, "train", train_metrics)
            log_metrics(csv_path, epoch, "val", val_metrics)
            mlflow_logger.log_metrics(train_metrics, step=epoch, prefix="train_")
            mlflow_logger.log_metrics(val_metrics, step=epoch, prefix="val_")
            print(
                f"epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.2f} "
                f"| val loss {val_metrics['loss']:.2f} (recon {val_metrics['reconstruction']:.2f}, "
                f"kl {val_metrics['kl']:.2f}, beta {beta:.4f})"
            )

            improved = val_metrics["loss"] < (state.best_loss - min_delta)
            if improved:
                state.best_loss = val_metrics["loss"]
                state.best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, state, config)
                mlflow_logger.log_metrics(
                    {
                        "best_val_loss": state.best_loss,
                        "best_epoch": state.best_epoch,
                        "best_val_reconstruction": val_metrics["reconstruction"],
                        "best_val_kl": val_metrics["kl"],
                    },
                    step=epoch,
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(model, optimizer, state, config, filename="last_checkpoint.pth")

            if early_stopping_enabled and epochs_without_improvement >= patience:
                print(
                    f"Arret anticipe : aucune amelioration de validation superieure a "
                    f"{min_delta} depuis {patience} epochs. Meilleure epoch : {state.best_epoch}."
                )
                break

        mlflow_logger.log_metrics({"best_val_loss": state.best_loss, "best_epoch": state.best_epoch})
        mlflow_logger.log_artifact(csv_path, artifact_path="logs")
        mlflow_logger.log_artifact(output_dir / "best_checkpoint.pth", artifact_path="checkpoints")
        mlflow_logger.log_artifact(output_dir / "last_checkpoint.pth", artifact_path="checkpoints")

    print(
        f"Entrainement termine. Meilleure loss de validation : {state.best_loss:.4f} "
        f"(epoch {state.best_epoch})"
    )
    return state
