"""Utilitaires MLflow optionnels pour le sous-projet CelebA.

Le code d'entrainement doit rester utilisable sans MLflow, par exemple pour
un smoke test rapide ou une machine ou la dependance n'est pas installee. Ce
module encapsule donc l'import de MLflow et expose une petite interface qui
devient silencieuse quand `mlflow.enabled` vaut false dans la config.
"""
from pathlib import Path
from typing import Any, Dict, Optional


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Aplatit une config imbriquee pour `mlflow.log_params`.

    MLflow accepte des valeurs simples comme parametres. Les listes sont
    converties en chaine lisible, et les sous-dicts deviennent des cles du
    type `training.lr`.
    """
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        elif isinstance(value, (list, tuple)):
            flat[full_key] = ",".join(str(item) for item in value)
        else:
            flat[full_key] = value
    return flat


class MLflowLogger:
    """Petit wrapper qui rend MLflow optionnel pour le trainer."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.mlflow = None
        self.active = False

    def __enter__(self) -> "MLflowLogger":
        mlflow_cfg = self.config.get("mlflow", {})
        if not mlflow_cfg.get("enabled", False):
            return self

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow est active dans la config, mais le package 'mlflow' "
                "n'est pas installe. Lancez `pip install -r requirements.txt` "
                "dans projects/blaise_celeba/."
            ) from exc

        self.mlflow = mlflow
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = mlflow_cfg.get("experiment_name", "blaise_celeba")
        mlflow.set_experiment(experiment_name)
        run_name = mlflow_cfg.get("run_name") or self._default_run_name()
        mlflow.start_run(run_name=run_name)
        self.active = True

        tags = {
            "dataset": self.config.get("dataset", {}).get("name", "celeba"),
            "model_type": self.config.get("model", {}).get("type", "vae"),
            "project": "blaise_celeba",
        }
        tags.update(mlflow_cfg.get("tags", {}))
        mlflow.set_tags(tags)
        self.log_params(flatten_dict({k: v for k, v in self.config.items() if k != "mlflow"}))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.active and self.mlflow is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.mlflow.end_run(status=status)
        self.active = False

    def _default_run_name(self) -> str:
        model_type = self.config.get("model", {}).get("type", "vae")
        output_dir = Path(self.config.get("training", {}).get("output_dir", "run"))
        return f"{model_type}-{output_dir.name}"

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.active or self.mlflow is None:
            return
        # MLflow limite la taille des valeurs de parametres ; on tronque sans
        # perdre l'information utile pour les configs de ce projet.
        safe_params = {key: str(value)[:500] for key, value in params.items()}
        self.mlflow.log_params(safe_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        if not self.active or self.mlflow is None:
            return
        named_metrics = {
            f"{prefix}{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        self.mlflow.log_metrics(named_metrics, step=step)

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.active or self.mlflow is None or not path.exists():
            return
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not self.active or self.mlflow is None or not path.exists():
            return
        self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)
