import argparse
import os
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._config = config_dict

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._config


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cast_scalar(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if "." in value or "e" in value or "E" in value:
                return float(value)
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def merge_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    result = dict(config)
    for key, value in vars(args).items():
        if value is not None:
            result[key] = value

    if "training" in result:
        for key in ["lr", "beta", "epochs", "batch_size", "seed"]:
            if key in result["training"]:
                result["training"][key] = cast_scalar(result["training"][key])

    if "model" in result:
        for key in ["latent_dim", "hidden_channels"]:
            if key in result["model"]:
                result["model"][key] = cast_scalar(result["model"][key])

    return result


def build_config() -> Config:
    parser = argparse.ArgumentParser(description="Chargement de la configuration YAML")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("--seed", type=int, help="Seed de reproductibilite\n")
    parser.add_argument("--batch-size", type=int, help="Taille des lots")
    parser.add_argument("--lr", type=float, help="Taux d'apprentissage")
    parser.add_argument("--epochs", type=int, help="Nombre d'epochs")
    parser.add_argument("--beta", type=float, help="Coefficient beta pour la loss ELBO")
    parser.add_argument("--device", type=str, help="Appareil d'execution: cpu ou cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Limite le run pour un test rapide")
    args, _ = parser.parse_known_args()
    config = load_yaml_config(args.config)
    merged = merge_args(config, args)
    return Config(merged)
