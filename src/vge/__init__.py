"""
Variance-Gated Ensembles (VGE)
A framework for epistemic-aware uncertainty estimation in deep ensembles.
"""

from .metrics import compute_epjs, compute_epkl, compute_vgmu, safe_decomposition
from .models import VGN, DeepEnsemble, LastLayerEnsemble, WideResNet
from .predictors import ensemble_predict, mcd_predict, predict, vgn_predict
from .trainers import EnsembleTrainer, Trainer, VGNTrainer
from .utils import EarlyStopping, select_device, set_seeds
from .vgn import VarianceGatedNormalizer

__version__ = "1.0.0"
__all__ = [
    # Core VGN
    "VarianceGatedNormalizer",
    # Models
    "WideResNet",
    "DeepEnsemble",
    "LastLayerEnsemble",
    "VGN",
    # Trainers
    "Trainer",
    "EnsembleTrainer",
    "VGNTrainer",
    # Predictors
    "predict",
    "ensemble_predict",
    "vgn_predict",
    "mcd_predict",
    # Metrics
    "safe_decomposition",
    "compute_epkl",
    "compute_epjs",
    "compute_vgmu",
    # Utilities
    "select_device",
    "set_seeds",
    "EarlyStopping",
]
