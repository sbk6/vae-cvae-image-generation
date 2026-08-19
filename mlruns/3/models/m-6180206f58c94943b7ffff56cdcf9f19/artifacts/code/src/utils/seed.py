import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixe les seeds pour rendre les expériences reproductibles.

    On fixe le seed de Python, NumPy et PyTorch, et on demande à PyTorch
    de désactiver les algorithmes nondéterministes si possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
