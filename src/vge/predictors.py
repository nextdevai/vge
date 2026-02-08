"""
Prediction utilities for Variance-Gated Ensembles.

Contains inference functions for:
- predict: Baseline single-model inference
- ensemble_predict: Ensemble model inference
- vgn_predict: VGN model inference with uncertainty estimates
- mcd_predict: Monte Carlo Dropout inference

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .models import VGN


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    fname: str = "testing_results.npz",
) -> None:
    """
    Run inference with a baseline (non-ensemble) model.

    Saves results to an .npz file with keys:
        - L: True labels (N,)
        - P: Softmax probabilities (N, C)

    Args:
        model: Baseline model with single output head
        dataloader: DataLoader for inference
        device: Device to run inference on
        fname: Output file path for the .npz results
    """
    model.eval()

    all_L: list[np.ndarray] = []
    all_P: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            values = [*batch]
            inputs, labels = values[0].to(device), values[1].to(device)

            logits: Tensor = model(inputs)
            probs = F.softmax(logits, dim=-1)

            all_L.append(labels.cpu().numpy())
            all_P.append(probs.cpu().numpy())

    L = np.concatenate(all_L, axis=0)  # (N,)
    P = np.concatenate(all_P, axis=0)  # (N, C)

    np.savez(fname, L=L, P=P)

    return None


def vgn_predict(
    model: VGN,
    dataloader: DataLoader,
    device: str | torch.device,
    fname: str = "testing_results.npz",
) -> None:
    """
    Run inference with a VGN model.

    Collects predictions, uncertainty estimates, and labels for all
    samples in the dataloader.

    Saves results to an .npz file with keys:
        - L: True labels (N,)
        - Q: Gated probabilities (N, M, C)
        - k: Gate parameters

    Args:
        model: VGN model
        dataloader: DataLoader for inference
        device: Device to run inference on
        fname: Output file path for the .npz results
    """
    model.eval()

    all_L: list[np.ndarray] = []
    all_Q: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            values = [*batch]
            inputs, labels = values[0].to(device), values[1].to(device)

            probs: Tensor
            probs, _ = model(inputs)

            all_L.append(labels.cpu().numpy())
            all_Q.append(probs.cpu().numpy())

    L = np.concatenate(all_L, axis=0)  # (N,)
    Q = np.concatenate(all_Q, axis=0)  # (N, M, C)
    k = model.vgn.k_value().detach().cpu().numpy()  # (C,)

    np.savez(fname, L=L, Q=Q, k=k)

    return None


def ensemble_predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    fname: str = "testing_results.npz",
) -> None:
    """
    Run inference with an ensemble model (without VGN).

    Saves results to an .npz file with keys:
        - L: True labels (N,)
        - P: Ensemble probabilities (N, M, C)

    Args:
        model: Ensemble model (e.g., LastLayerEnsemble or DeepEnsemble)
        dataloader: DataLoader for inference
        device: Device to run inference on
        fname: Output file path for the .npz results
    """
    model.eval()

    all_L: list[np.ndarray] = []
    all_P: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            values = [*batch]
            inputs, labels = values[0].to(device), values[1].to(device)

            probs: Tensor = model(inputs)

            all_P.append(probs.cpu().numpy())
            all_L.append(labels.cpu().numpy())

    L = np.concatenate(all_L, axis=0)  # (N,)
    P = np.concatenate(all_P, axis=0)  # (N, M, C)

    np.savez(fname, L=L, P=P)

    return None


@torch.no_grad()
def mcd_predict(
    model: nn.Module,
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    device: str | torch.device,
    dropout_prob: float = 0.1,
    num_heads: int = 1,
    num_samples: int = 10,
    seed: int = 0,
    last_layer: bool = False,
    fname: str = "testing_probabilities.npz",
) -> None:
    """
    Run Monte Carlo Dropout inference for uncertainty estimation.

    Supports two modes:
    - Standard MCD (last_layer=False): Dropout throughout the network
    - Last-layer MCD (last_layer=True): Dropout only in classifier heads

    The model must have a `classifiers` attribute (nn.ModuleList) for
    multihead support, and an `mlp` attribute for feature extraction.

    Saves results to an .npz file with keys:
        - L: True labels (N,)
        - P: MC Dropout probabilities (N, num_samples * num_heads, C)

    Args:
        model: Base model architecture (will be copied, not modified)
        checkpoint_path: Path to model checkpoint with 'model_state_dict' key
        dataloader: DataLoader for inference
        device: Device to run inference on
        dropout_prob: Dropout probability during inference
        num_heads: Number of classifier heads (duplicates single head if needed)
        num_samples: Number of stochastic forward passes per head
        seed: Random seed for reproducibility
        last_layer: If True, only apply dropout to classifier heads
        fname: Output file path for the .npz results
    """

    def _configure_model(
        base_model: nn.Module,
        ckpt_path: str | Path,
        p: float,
        dev: str | torch.device,
        ll_only: bool,
    ) -> nn.Module:
        """Configure model for MC Dropout inference."""
        m = copy.deepcopy(base_model).to(dev)

        ckpt = torch.load(ckpt_path, map_location=dev)
        state = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state, strict=False)

        # Turn off all dropout initially
        for mod in m.modules():
            if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
                mod.p = 0.0

        if ll_only:
            # Only classifier heads get dropout
            if not hasattr(m, "classifiers") or not isinstance(
                m.classifiers, nn.ModuleList
            ):
                raise AttributeError(
                    "Model must have `classifiers` as nn.ModuleList for last_layer mode."
                )
            for head in m.classifiers:
                for mod in head.modules():
                    if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
                        mod.p = p
        else:
            # Dropout everywhere
            for mod in m.modules():
                if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
                    mod.p = p

        # Enable training mode so dropout is active
        m.train()

        # Freeze BatchNorm statistics
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()

        return m

    def _configure_heads(m: nn.Module, target_heads: int) -> None:
        """Ensure model has the required number of classifier heads."""
        if not hasattr(m, "classifiers") or not isinstance(
            m.classifiers, nn.ModuleList
        ):
            raise AttributeError("Model must have `classifiers` as nn.ModuleList.")
        if len(m.classifiers) == 0:
            raise AttributeError("Classifiers must contain at least one head.")
        if target_heads > 1 and len(m.classifiers) == 1:
            base_head = m.classifiers[0]
            m.classifiers = nn.ModuleList(
                [copy.deepcopy(base_head) for _ in range(int(target_heads))]
            )
        assert len(m.classifiers) == target_heads, (
            f"Expected {target_heads} classifier heads, but found {len(m.classifiers)}."
        )

    def _get_mlp_features(mdl: nn.Module, x: Tensor, mlp_attr: str = "mlp") -> Tensor:
        """Extract features from the MLP layer using a forward hook."""
        if not hasattr(mdl, mlp_attr):
            raise AttributeError(f"Model has no attribute '{mlp_attr}'.")
        mlp_module = getattr(mdl, mlp_attr)
        mlp_out: dict[str, Tensor] = {}

        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], out: Tensor) -> None:
            mlp_out["feat"] = out.detach()

        handle = mlp_module.register_forward_hook(hook)
        _ = mdl(x)
        handle.remove()
        return mlp_out["feat"]

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configure model
    base = _configure_model(
        model, checkpoint_path, p=dropout_prob, dev=device, ll_only=last_layer
    )
    _configure_heads(base, int(num_heads))
    classifiers: nn.ModuleList = getattr(base, "classifiers")

    H = int(num_heads)
    S = int(num_samples)
    assert S >= 1 and H >= 1

    all_L: list[np.ndarray] = []
    all_P: list[np.ndarray] = []

    for batch in dataloader:
        values = [*batch]
        batch_inputs, batch_labels = values[0].to(device), values[1].to(device)
        all_L.append(batch_labels.cpu().numpy())

        sample_probs_list: list[Tensor] = []

        if last_layer:
            # Backbone is deterministic; compute features once
            features_fixed = _get_mlp_features(base, batch_inputs)

            for s in range(S):
                torch.manual_seed(seed + 1 + s)
                np.random.seed(seed + 1 + s)

                logits_list = [head(features_fixed) for head in classifiers]
                batch_logits = torch.stack(logits_list, dim=1)  # [B, H, C]
                batch_probs = F.softmax(batch_logits, dim=-1)

                sample_probs_list.append(batch_probs.unsqueeze(1))  # [B, 1, H, C]
        else:
            # Full-network MC dropout: recompute features each sample
            for s in range(S):
                torch.manual_seed(seed + 1 + s)
                np.random.seed(seed + 1 + s)

                features_s = _get_mlp_features(base, batch_inputs)
                logits_list = [head(features_s) for head in classifiers]
                batch_logits = torch.stack(logits_list, dim=1)  # [B, H, C]
                batch_probs = F.softmax(batch_logits, dim=-1)

                sample_probs_list.append(batch_probs.unsqueeze(1))  # [B, 1, H, C]

        sample_probs: Tensor = torch.cat(sample_probs_list, dim=1)  # [B, S, H, C]

        B, S_, H_, C = sample_probs.shape
        assert S_ == S and H_ == H

        sample_probs = sample_probs.reshape(B, S * H, C)  # [B, S * H, C]
        all_P.append(sample_probs.cpu().numpy())

    L = np.concatenate(all_L, axis=0)  # [N,]
    P = np.concatenate(all_P, axis=0)  # [N, S * H, C]

    np.savez(fname, L=L, P=P)

    return None
