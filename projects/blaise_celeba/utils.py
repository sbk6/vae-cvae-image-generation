"""Petits utilitaires partages par les scripts d'entrainement et d'evaluation."""
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def set_seed(seed: int) -> None:
    """Fixe les seeds Python/NumPy/PyTorch pour rendre un run reproductible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
